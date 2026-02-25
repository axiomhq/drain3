# Drain3: Template-Based Log Compression

## Overview

This library repurposes the [Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf) log parsing algorithm for **compression**. The core insight: most log lines in a dataset are structurally identical, differing only in a few variable fields (timestamps, IDs, counts, etc.). Instead of storing every line verbatim, we store a set of **templates** once and then represent each line as either a `(template_id, params...)` tuple or a raw fallback.

## The 5-Step Compression Flow

### Step 1: Sample rows for training

Take a random sample of the full log dataset (e.g. 10%) and feed it to `Train()` / `TrainWithConfig()`. The sample must be large enough to discover the dominant patterns but small enough to keep training fast. The benchmark uses a 10% sample with a fixed seed for reproducibility.

```go
rng := rand.New(rand.NewSource(42))
var sample []string
for _, line := range allLines {
    if rng.Float64() < 0.10 {
        sample = append(sample, line)
    }
}
matcher, _ := Train(sample)
```

**What training does internally:**

For each sample line, Drain tokenizes it (whitespace-split), then searches a prefix tree for a cluster whose template is similar enough (`SimilarityThreshold`, default 0.4). If found, the line is merged into that cluster -- any token positions where the line and template disagree are replaced with the wildcard `<*>`. If no match is found, a new cluster is created with the line's tokens as a fresh template.

After all samples are processed, the mutable tree is frozen into an optimized flat tree (sorted arrays + binary search), a prefilter index is built on first/last tokens, and the matcher is ready.

### Step 2: Build a compact template ID space

`matcher.Templates()` returns clusters with sparse, internal IDs (1, 3, 7, ...). For columnar compression you want a dense `[0, N)` ID space. Build a remap table:

```go
templates := matcher.Templates()
clusterToTemplate := make([]int, maxClusterID+1)  // sparse → dense
for i := range clusterToTemplate {
    clusterToTemplate[i] = -1  // sentinel
}
paramString := matcher.Config().ParamString
templateParamCounts := make([]int, len(templates))
for denseID, t := range templates {
    clusterToTemplate[t.ID] = denseID
    templateParamCounts[denseID] = countParams(t.Tokens, paramString)
}
```

Each template knows how many `<*>` slots it has, so you know exactly how many params to expect per matched line.

### Step 3: Match each line

For each line in the **full** dataset, call `MatchInto(line, scratch[:0])`:

```go
scratch := make([]string, 0, 64)
for _, line := range allLines {
    clusterID, params, matched := matcher.MatchInto(line, scratch[:0])
    // ...
}
```

`MatchInto` reuses the scratch buffer to avoid allocations. It returns:
- `clusterID`: the internal cluster ID (use the remap table from step 2)
- `params`: the extracted variable tokens, in order of `<*>` positions
- `matched`: whether the line fit any template

**What matching does internally:**

1. Fast reject: count spaces to get token count; reject if no templates exist with that length.
2. Prefilter (if enabled): look up first/last token in edge indexes to narrow candidates.
3. Tree search: walk the flat prefix tree following token IDs (or the wildcard node).
4. Score candidates: for each candidate cluster, compare non-`<*>` positions. Require `score >= MatchThreshold * tokenCount` (default threshold = 1.0, meaning **all** non-param tokens must match exactly).
5. Extract params: walk the template, collect input tokens at `<*>` positions.

### Step 4: Store matched vs. unmatched

- **If matched**: store `(templateID, param_0, param_1, ...)` -- the dense template ID from step 2, plus each extracted parameter. This is the compressed representation.
- **If not matched**: store the raw line verbatim. This is the fallback that guarantees lossless round-trip.

```go
if matched {
    denseID := clusterToTemplate[clusterID]
    // store denseID + params in columnar format
} else {
    // store raw line
}
```

The compression ratio depends on how repetitive the logs are. If 1M lines cluster into 500 templates, you store 500 template strings once and then 1M small tuples of `(uint16 + a few param strings)` instead of 1M full lines.

### Step 5: Reconstruct from template + param streams

To decompress, join the template tokens back together, substituting `<*>` with the stored params in order:

```go
func reconstruct(tmpl Template, params []string, paramString string) string {
    result := make([]string, len(tmpl.Tokens))
    pi := 0
    for i, tok := range tmpl.Tokens {
        if tok == paramString {
            result[i] = params[pi]
            pi++
        } else {
            result[i] = tok
        }
    }
    return strings.Join(result, " ")
}
```

For raw fallback rows, emit the stored string as-is.

## Why This Compresses Well

| What's stored | Without compression | With template compression |
|---|---|---|
| 1M identical log structures | 1M full strings | 1 template + 1M param tuples |
| Template overhead | 0 | N templates (typically 100s-1000s) |
| Unique / rare lines | 1 string each | 1 string each (raw fallback) |

The params themselves are often short (numbers, UUIDs, status codes) vs. the full log line which includes all the static boilerplate. In a columnar format, template IDs compress further since they repeat heavily.

## Matcher Serialization

The trained matcher serializes to a compact binary format (`MarshalBinary` / `LoadMatcher`) so you can train once and reuse across decode operations:

```
"drn3" magic (4 bytes)
version byte (0x01)
config (depth, thresholds, max_children, param_string, flags, delimiters)
template_count (uvarint)
for each template:
    id (varint), count (varint), token_count (uvarint)
    tokens... (length-prefixed strings)
```

On load, all runtime indexes (prefix tree, prefilter, match thresholds) are rebuilt from the templates.

## Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `SimilarityThreshold` | 0.4 | How aggressively to merge during **training** (lower = fewer, broader templates) |
| `MatchThreshold` | 1.0 | How strict matching is at **compression time** (1.0 = all non-param tokens must match) |
| `Depth` | 4 | Prefix tree depth (controls how many leading tokens are indexed) |
| `MaxChildren` | 100 | Max children per tree node before forcing wildcard |
| `ParametrizeNumericTokens` | true | Auto-wildcard tokens containing digits during training |
| `EnableMatchPrefilter` | true | Use first/last token index to speed up matching |
| `ExtraDelimiters` | `[]` | Additional characters to split on besides whitespace |

**Key tradeoff**: `SimilarityThreshold` controls training generalization. Lower values create fewer, broader templates (better compression ratio, but risk merging unrelated patterns). `MatchThreshold` at 1.0 ensures lossless reconstruction -- a line only matches if every fixed token is identical.
