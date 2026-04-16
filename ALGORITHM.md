# The Drain Algorithm

Reference for how this library implements the [Drain log template
parser](https://pinjiahe.github.io/papers/ICWS17.pdf). This document
describes data structures and the two hot paths (training, matching),
and is the place to look when you want to change behavior rather than
use it. For usage, see `README.md`.

## What Drain does

Drain incrementally groups log lines into **clusters**. Each cluster
has a **template** — a token sequence with `<*>` placeholders at
positions that varied across its member lines. The algorithm is online
(processes one line at a time), approximate, and optimized around one
empirical assumption: for most log formats, the **first few tokens
identify the shape**, so they can be used as a prefix-tree index.

A typical outcome: 2M lines collapse to a few hundred templates.

## Data structures

### Prefix tree

Depth `D` (default 4) means `D` levels, where the cluster-IDs live at
the bottom:

```
level 1  root
level 2    grouped by token count
level 3      grouped by first token
...
level D        leaf: list of cluster IDs sharing this prefix
```

Each internal node is a `map[uint64]*node` (child key = interned token
ID, or `paramID` for the wildcard sibling). Each leaf keeps a
`[]int` of cluster IDs rooted there.

### Cluster

A cluster stores the template token sequence twice for fast matching:

- `tokenIDs []uint64` — token IDs for tree/numeric comparisons.
- `tokenStr []string` — token strings for direct comparison (prefilter path).
- `nonParamIdx []uint16` — indexes of non-`<*>` positions (skip wildcards in scoring loops).
- `anchor0`, `anchor1` — first and last non-param position indexes, used for cheap pre-rejection before the full scan.

### Token dictionary

During training, tokens are interned into `dictIDs map[string]uint64`.
After training, `freezeDict()` copies the dictionary into
[constmap](https://github.com/lemire/constmap), a verified perfect-hash
map, for faster probes on the match path. Measured impact: ~5 % on
high-volume match workloads (~2M lines, mixed hit/miss).

### Prefilter index

Keyed by `tokenCount`, each bucket has four lookup tables mapping token
IDs to candidate cluster IDs:

- `any` — templates whose first AND last positions are params.
- `firstKeys`/`firstVals` — templates whose first position is a concrete token.
- `lastKeys`/`lastVals` — templates whose last position is a concrete token.
- `flKeys`/`flVals` — templates whose both edges are concrete (first+last packed into a single `uint64`).

The prefilter narrows candidates using only 2 dict probes (first/last
token) before string-comparing against each candidate's full template.

## Training: `Train(lines)` → `*Matcher`

For each line, `addLogMessage` does:

1. **Tokenize**: split on single spaces (preserves empty tokens from
   consecutive spaces, so lossless reconstruction is possible).
2. **Reject**: skip lines longer than `MaxBytes` or more than `MaxTokens` tokens.
3. **Tree search** with `SimilarityThreshold`:
   - Walk the tree to the leaf for this token count + first token.
   - Score each candidate cluster's template against the line:
     similarity = count of matching non-param positions ÷ token count.
   - Tie-break: more params wins (prefer more general templates).
4. **If a candidate scores ≥ threshold**: merge. Walk template and line
   together; positions that disagree become `<*>`. Rebuild
   `nonParamIdx` / anchors.
5. **Otherwise**: create a new cluster. Intern tokens, insert into the
   tree creating internal nodes as needed (see routing rules below).

After all lines: `syncTemplatesFromClusters` builds the public
`Templates()` view, `rebuildMatchPrefilter` builds the prefilter index,
`rebuildMatchNeeded` precomputes the required-score table, and
`freezeDict` swaps the growing map for the frozen one.

### Routing rules when inserting a new token at an internal node

- If `ParametrizeNumericTokens` is true and the token contains any
  digit, route through the wildcard child (digits are usually
  variable). Create the wildcard child if needed.
- Otherwise, if a wildcard sibling already exists, give specific
  children up to `MaxChildren - 1` slots. After that, new tokens route
  through the wildcard.
- If no wildcard yet, create specific children until
  `len(children) == MaxChildren - 1`, at which point the next unseen
  token becomes the reserved wildcard; all further unseen tokens route
  through it.

### `MaxClusters` cap

When `MaxClusters > 0` and a new cluster would exceed the cap, the
line is dropped (no cluster created, no update). This differs from the
Python reference which uses LRU eviction.

## Matching: `Match(line)` and `MatchInto(line, buf)`

Matching requires a **perfect** match: at the default
`MatchThreshold` of 1.0, every non-param position must agree.

```
Match(line)       = MatchInto(line, nil)
MatchID(line)     = same as Match but skips arg extraction
MatchInto(line, buf) = full API; writes extracted args into buf
```

`findMatch` does:

1. **Unknown-first-token reject**: if no template has a param at
   position 0, and the line's first token isn't in the frozen dict, no
   cluster can match — return nil immediately. (This is the reason
   `BenchmarkTrainMatch/match_miss` is ~10× faster than the hit path.)
2. **Tokenize and length-reject**: skip if token count exceeds
   `MaxTokens` or no cluster has this token count.
3. **Fast path (prefilter)**: combine `any` / `first` / `last` /
   `first+last` candidate buckets for this token count. Score each
   candidate by direct string comparison against its `tokenStr`, with
   anchor pre-rejection (first and last non-param positions) before
   the full non-param loop. At threshold 1.0 we bail on first
   mismatch and return the first perfect match.
4. **Slow path (tree search)**: used when the prefilter is disabled.
   Resolve tokens to IDs via the frozen dict, walk the tree, score at
   the leaf.
5. **Extract args**: `extractArgsInto` walks the template's
   `tokenIDs`, appending input tokens at each `paramID` position into
   the caller's scratch buffer. The hit path is zero-alloc when the
   buffer has enough capacity.

## Configuration

See `config.go` for the authoritative definitions. The subset that
shapes behavior most:

| Field | Default | Effect |
| --- | --- | --- |
| `Depth` | 4 | Tree depth. Must be ≥ 3. |
| `SimilarityThreshold` | 0.5 | Training merge threshold. Lower = fewer, broader templates. |
| `MatchThreshold` | 1.0 | Match-time threshold. 1.0 = every non-param must agree. |
| `MaxChildren` | 100 | Fanout cap per internal node before wildcard promotion. |
| `MaxClusters` | 0 | Cluster cap (0 = unlimited; on cap, lines are dropped). |
| `MaxTokens` | 64 | Reject lines with more tokens. |
| `MaxBytes` | 1024 | Reject longer lines. |
| `ParamString` | `<*>` | Placeholder token. |
| `ParametrizeNumericTokens` | true | Route digit-bearing tokens through the wildcard child. |
| `EnableMatchPrefilter` | true | Use edge-token index instead of tree descent at match time. |
| `ExtraDelimiters` | `nil` | Additional substrings to replace with spaces before splitting. |

### Tuning notes

- **Training vs matching thresholds.** `SimilarityThreshold` decides
  how aggressively training merges. `MatchThreshold` decides how
  strict matching is at query time. They're independent: you can
  train permissively (0.4) and still require exact matches (1.0).
- **`ParametrizeNumericTokens` and first tokens.** If your first
  token is usually numeric (e.g. a timestamp leads every line), this
  routes every line through the wildcard child at level 2. You still
  cluster by the *next* tokens, but the first-token prefilter becomes
  useless — consider setting it to false or using `ExtraDelimiters` to
  reshape the line.
