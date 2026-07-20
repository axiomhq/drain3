# Batch Match API + SIMD Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Session-based batch match API (SoA results, shareable trained Matcher) and swap the tokenizer's space scan for platform SIMD kernels (NEON/AVX2, Plan 9 asm) with the SWAR loop as fallback and oracle.

**Architecture:** `Session` takes ownership of all per-call scratch so a trained `Matcher` is immutable and shareable; `MatchBatch` writes struct-of-arrays results into a reusable arena. The tokenizer splits into a `spaceBitmap` scan kernel (build-tag selected: NEON on arm64, AVX2 on amd64, pure-Go SWAR elsewhere) and one shared emission loop.

**Tech Stack:** Go 1.25+, Plan 9 assembly (`.s` files), `golang.org/x/sys/cpu` (Task 5 only, AVX2 detection).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-batch-api-simd-tokenizer-design.md`. Work order and gates are binding.
- Match results must stay byte-identical: every task ends with the full test suite green (`go test -count=1 .`).
- Warm paths allocate nothing: `MatchBatch` with a reused `BatchResult` and per-line `MatchInto` must report 0 allocs/op in benchmarks.
- Assembly never loads past the end of a string (strings may end at a page boundary): asm processes only full 64-byte blocks; Go handles tails.
- Commit messages: plain, no Co-Authored-By/attribution footers (user preference).
- House style applies (`engineering-workflows:go-house-style`): stdlib-only except `golang.org/x/sys` in Task 5; one `_test.go` per `.go` file; table-driven stdlib tests, no testify.
- Existing public API keeps signatures and single-goroutine-per-Matcher semantics.
- `strs.json` (709MB corpus, untracked) may be absent: corpus tests/benches must skip cleanly when it is missing.
- Benchmarks comparing before/after must run the test binary from a directory WITHOUT strs.json (`go test -c` then run the binary from the scratch dir), so the corpus init-load doesn't skew GC.

---

### Task 1: Session type owning match scratch

**Files:**
- Create: `session.go`
- Create: `session_test.go`
- Modify: `drain.go` (Matcher fields, freezeDict), `match.go` (receiver conversions), `train.go` (no logic change; verify build)
- Test: `session_test.go`, existing suite

**Interfaces:**
- Consumes: existing `Matcher` internals (`cfg`, `dictFrozen`, `rootByLen`, `prefilterBuckets`, `maxProbe`, scoring methods).
- Produces (later tasks rely on these exact names):
  - `type Session struct { m *Matcher; tok []string; candidates []int; probeIDs []uint64 }`
  - `func (m *Matcher) NewSession() *Session`
  - `func (s *Session) Match(line string) (templateID int, args []string, ok bool)`
  - `func (s *Session) MatchID(line string) (templateID int, ok bool)`
  - `func (s *Session) MatchInto(line string, dst []string) (templateID int, args []string, ok bool)`
  - `func (s *Session) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool)`
  - `func (s *Session) findMatch(line string) (*cluster, []string)` and `func (s *Session) findExactMatch(line string) (*cluster, []string)` (internal, Task 2 uses `findMatch`)

- [ ] **Step 1: Write the failing tests**

Append to a new `session_test.go` (package `drain3`, internal — mirrors drain_test.go imports):

```go
package drain3

import (
	"sync"
	"testing"
)

// TestSessionMatchesMatcher pins Session methods to the Matcher methods
// they replace: identical results for hits, misses, and arg extraction.
func TestSessionMatchesMatcher(t *testing.T) {
	m, err := Train([]string{
		"svc auth user 1 ip 10.0.0.1",
		"svc auth user 2 ip 10.0.0.2",
		"cache hit key foo",
		"cache miss key bar",
	})
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	s := m.NewSession()
	queries := []string{
		"svc auth user 9 ip 1.2.3.4", // hit with args
		"cache hit key baz",          // hit
		"completely unknown line",    // miss
		"",                           // empty
	}
	for _, q := range queries {
		mID, mArgs, mOK := m.Match(q)
		sID, sArgs, sOK := s.Match(q)
		if mID != sID || mOK != sOK || len(mArgs) != len(sArgs) {
			t.Fatalf("Match(%q): matcher=(%d,%v,%v) session=(%d,%v,%v)", q, mID, mArgs, mOK, sID, sArgs, sOK)
		}
		for i := range mArgs {
			if mArgs[i] != sArgs[i] {
				t.Fatalf("Match(%q) arg %d: %q vs %q", q, i, mArgs[i], sArgs[i])
			}
		}
		mID, mOK = m.MatchID(q)
		sID, sOK = s.MatchID(q)
		if mID != sID || mOK != sOK {
			t.Fatalf("MatchID(%q): matcher=(%d,%v) session=(%d,%v)", q, mID, mOK, sID, sOK)
		}
		eID, _, eOK := m.MatchExactInto(q, nil)
		xID, _, xOK := s.MatchExactInto(q, nil)
		if eID != xID || eOK != xOK {
			t.Fatalf("MatchExactInto(%q): matcher=(%d,%v) session=(%d,%v)", q, eID, eOK, xID, xOK)
		}
	}
}

// TestSessionConcurrent shares one trained Matcher across goroutines,
// one Session each. Run with -race.
func TestSessionConcurrent(t *testing.T) {
	m, err := Train([]string{
		"svc auth user 1 ip 10.0.0.1",
		"svc auth user 2 ip 10.0.0.2",
		"cache hit key foo",
	})
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := m.NewSession()
			for i := 0; i < 1000; i++ {
				if _, ok := s.MatchID("svc auth user 7 ip 9.9.9.9"); !ok {
					t.Error("expected hit")
					return
				}
				if _, ok := s.MatchID("zzz unknown zzz"); ok {
					t.Error("expected miss")
					return
				}
			}
		}()
	}
	wg.Wait()
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -run 'TestSession' -count=1 .`
Expected: FAIL — `m.NewSession undefined`.

- [ ] **Step 3: Create session.go**

```go
package drain3

// Session holds the mutable per-call state for matching. A trained
// Matcher is immutable; any number of goroutines may share it, one
// Session per goroutine. A Session must not be used concurrently.
type Session struct {
	m          *Matcher
	tok        []string
	candidates []int
	probeIDs   []uint64
}

// NewSession returns a Session for matching against m. It panics if m
// has not been trained (built via Train, TrainWithConfig, or
// NewMatcherFromTemplates).
func (m *Matcher) NewSession() *Session {
	if m == nil || m.dictFrozen == nil {
		panic("drain3: NewSession on an untrained Matcher")
	}
	return &Session{
		m:          m,
		tok:        make([]string, 0, m.cfg.MaxTokens),
		candidates: make([]int, 0, 1024),
		probeIDs:   make([]uint64, m.maxProbe),
	}
}

// Match returns template id, extracted args, and whether a match was found.
func (s *Session) Match(line string) (templateID int, args []string, ok bool) {
	return s.MatchInto(line, nil)
}

// MatchID returns just the template id and whether a match was found.
func (s *Session) MatchID(line string) (templateID int, ok bool) {
	cluster, _ := s.findMatch(line)
	if cluster == nil {
		return 0, false
	}
	return cluster.id, true
}

// MatchInto returns template id, extracted args into dst, and whether a
// match was found.
func (s *Session) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := s.findMatch(line)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, s.m.paramID, dst), true
}

// MatchExactInto returns a match only when every non-param template token
// exactly matches the input token at the same position; param positions
// act as wildcards. Ties break to the most parametrized template, then
// lowest template ID.
func (s *Session) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := s.findExactMatch(line)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, s.m.paramID, dst), true
}
```

- [ ] **Step 4: Move scratch off Matcher in drain.go**

In the `Matcher` struct, replace the scratch block:

```go
	// Scratch buffers — reused across calls.
	scratchIDs        []uint64
	scratchTok        []string
	scratchCandidates []int
	scratchProbeIDs   []uint64 // per-line anchor-position ID resolutions, indexed by probePos slot
```

with:

```go
	maxProbe int // widest probePos across prefilter buckets; sizes Session.probeIDs

	// Training-only scratch. Match-path scratch lives on Session.
	scratchIDs []uint64
	scratchTok []string

	// defaultSession backs the Matcher-level Match* methods, preserving
	// the historical one-goroutine-per-Matcher contract for them.
	defaultSession *Session
```

In `freezeDict`, delete the `m.scratchCandidates = make(...)` line and keep the `scratchTok` sizing (training still uses it). At the END of `freezeDict` (it is the last build step), add:

```go
	m.defaultSession = m.NewSession()
```

In `rebuildMatchPrefilter` (match.go), replace `m.scratchProbeIDs = make([]uint64, maxProbe)` with `m.maxProbe = maxProbe`.

- [ ] **Step 5: Convert the match entry points in match.go to Session methods**

Delete the Matcher-level bodies of `Match`, `MatchID`, `MatchInto`, `MatchExactInto` and replace with delegators (these stay in match.go):

```go
// Match returns template id, extracted args, and whether a match was found.
// Matcher-level Match* methods share one internal Session and follow the
// one-goroutine-per-Matcher rule; use NewSession for concurrent matching.
func (m *Matcher) Match(line string) (templateID int, args []string, ok bool) {
	return m.MatchInto(line, nil)
}

// MatchID returns just the template id and whether a match was found, without extracting args.
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	if m == nil || m.defaultSession == nil {
		return 0, false
	}
	return m.defaultSession.MatchID(line)
}

// MatchInto returns template id, extracted args into dst, and whether a match was found.
func (m *Matcher) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	if m == nil || m.defaultSession == nil {
		return 0, nil, false
	}
	return m.defaultSession.MatchInto(line, dst)
}

// MatchExactInto returns a match only when every non-param template token
// exactly matches the input token at the same position; param positions act
// as wildcards. When several templates qualify, the most parametrized one is
// returned, ties broken by lowest template ID, so the result is deterministic.
func (m *Matcher) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool) {
	if m == nil || m.defaultSession == nil {
		return 0, nil, false
	}
	return m.defaultSession.MatchExactInto(line, dst)
}
```

Convert these four internals from `(m *Matcher)` to `(s *Session)` receivers. Mechanical rule: immutable state reads become `s.m.<field>` (`cfg`, `dictFrozen`, `rootByLen`, `prefilterBuckets`, `hasParamFirst`, `paramID`); scoring/tree calls become `s.m.<method>(...)`; scratch becomes `s.<field>`.

`findMatch` (drop the `tokenBuf` parameter — it was always the scratch token buffer):

```go
func (s *Session) findMatch(line string) (cluster *cluster, tokens []string) {
	tokens, tokenCount, firstID, ok := s.tokenizeMatchLine(line)
	if !ok {
		return nil, nil
	}
	m := s.m
	// Fast path: prefilter narrows by token count + first/last token, then
	// scores candidates by string comparison, or — when the candidate set
	// is large enough to amortize resolving all tokens to dictionary IDs —
	// by uint64 comparison (see useIDScorerFor).
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := s.candidates
		if candidateIDs, owned, ok := m.prefilterCandidatesCompact(tokens, tokenCount, firstID, buf[:0]); ok {
			if owned && cap(candidateIDs) > cap(buf) {
				s.candidates = candidateIDs[:0:cap(candidateIDs)]
			}
			if m.useIDScorerFor(len(candidateIDs), tokenCount) {
				var idBuf [128]uint64
				ids := m.lookupTokenIDsPartial(tokens, idBuf[:0])
				return m.fastMatchIDs(candidateIDs, ids, m.cfg.MatchThreshold, true, false), tokens
			}
			return m.fastMatchStrings(candidateIDs, tokens, m.cfg.MatchThreshold, true, false), tokens
		}
		return nil, tokens
	}
	// Slow path: tree search needs token IDs for navigation.
	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.treeSearch(tokenIDs, m.cfg.MatchThreshold, true, false), tokens
}
```

`findExactMatch` follows the same conversion (`s.tokenizeMatchLine(line)`, `s.candidates`, `s.m.prefilterExactCandidatesCompact` becomes `s.prefilterExactCandidatesCompact` — see below, everything else via `m := s.m`).

`tokenizeMatchLine` becomes `func (s *Session) tokenizeMatchLine(line string) (tokens []string, tokenCount int, firstID uint64, ok bool)`: replace `m == nil` guard with nothing (Session always has a Matcher), `m.<field>` with `s.m.<field>`, and the `tokenBuf` parameter with `s.tok`.

`prefilterExactCandidatesCompact` becomes `func (s *Session) prefilterExactCandidatesCompact(tokens []string, tokenCount int, dst []int)`: `b := &s.m.prefilterBuckets[tokenCount]`, `dict := s.m.dictFrozen`, and `m.scratchProbeIDs` becomes `s.probeIDs`. `prefilterCandidatesCompact` uses no scratch and STAYS a Matcher method.

`treeSearchStrings` (training path) still uses `m.scratchTok` indirectly via addLogMessage — training code in train.go is untouched.

- [ ] **Step 6: Build and run the full suite**

Run: `go build ./... && go vet ./... && go test -race -count=1 .`
Expected: PASS (including the two new Session tests; -race clean).

- [ ] **Step 7: Verify per-line performance is unchanged**

```bash
SP=$(mktemp -d) && go test -c -o $SP/drain3.test . && cd $SP && ./drain3.test -test.run '^$' -test.bench 'BenchmarkTrainMatch/match_into$|BenchmarkLargeMixed' -test.benchtime 2s
```
Expected: match_into ≈ 490µs, LargeMixed/match_all ≈ 200ms (within run-to-run noise of the numbers in the previous commit message).

- [ ] **Step 8: Commit**

```bash
git add session.go session_test.go drain.go match.go
git commit -m "feat: Session type owns match scratch; trained Matcher is shareable

Matcher-level Match* methods keep their signatures and the historical
one-goroutine rule by delegating to an internal default Session."
```

---

### Task 2: BatchResult + MatchBatch

**Files:**
- Modify: `session.go` (BatchResult, MatchBatch), `drain.go` (cluster.appendArgs)
- Test: `session_test.go`, `match_test.go` (corpus equivalence, skips without strs.json)

**Interfaces:**
- Consumes: `(s *Session) findMatch(line string) (*cluster, []string)` from Task 1; `cluster.tokenIDs`, `Matcher.paramID`.
- Produces:
  - `type BatchResult struct { IDs []int32; ArgOff []int32; Args []string }`
  - `func (s *Session) MatchBatch(lines []string, dst *BatchResult) *BatchResult`
  - `func (c *cluster) appendArgs(dst []string, lineTokens []string, paramID uint64) []string`

- [ ] **Step 1: Write the failing tests**

Append to `session_test.go`:

```go
// TestMatchBatchEquivalence pins MatchBatch to per-line MatchInto.
func TestMatchBatchEquivalence(t *testing.T) {
	m, err := Train([]string{
		"svc auth user 1 ip 10.0.0.1",
		"svc auth user 2 ip 10.0.0.2",
		"cache hit key foo",
		"cache miss key bar",
	})
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	s := m.NewSession()
	lines := []string{
		"svc auth user 9 ip 1.2.3.4",
		"totally unknown shape",
		"cache hit key baz",
		"",
		"svc auth user 10 ip 8.8.8.8",
	}
	res := s.MatchBatch(lines, nil)
	if len(res.IDs) != len(lines) || len(res.ArgOff) != len(lines)+1 {
		t.Fatalf("shape: IDs=%d ArgOff=%d want %d/%d", len(res.IDs), len(res.ArgOff), len(lines), len(lines)+1)
	}
	if res.ArgOff[0] != 0 {
		t.Fatalf("ArgOff[0] = %d, want 0", res.ArgOff[0])
	}
	for i, line := range lines {
		id, args, ok := s.MatchInto(line, nil)
		gotArgs := res.Args[res.ArgOff[i]:res.ArgOff[i+1]]
		if !ok {
			if res.IDs[i] != 0 || len(gotArgs) != 0 {
				t.Fatalf("line %d %q: want miss, got id=%d args=%v", i, line, res.IDs[i], gotArgs)
			}
			continue
		}
		if int(res.IDs[i]) != id || len(gotArgs) != len(args) {
			t.Fatalf("line %d %q: batch=(%d,%v) perline=(%d,%v)", i, line, res.IDs[i], gotArgs, id, args)
		}
		for j := range args {
			if gotArgs[j] != args[j] {
				t.Fatalf("line %d arg %d: %q vs %q", i, j, gotArgs[j], args[j])
			}
		}
	}
}

// TestMatchBatchReuse verifies a warm BatchResult allocates nothing.
func TestMatchBatchReuse(t *testing.T) {
	m, err := Train([]string{
		"svc auth user 1 ip 10.0.0.1",
		"svc auth user 2 ip 10.0.0.2",
	})
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	s := m.NewSession()
	lines := []string{"svc auth user 9 ip 1.2.3.4", "nope", "svc auth user 3 ip 2.2.2.2"}
	var res BatchResult
	s.MatchBatch(lines, &res) // warm the arena
	allocs := testing.AllocsPerRun(100, func() {
		s.MatchBatch(lines, &res)
	})
	if allocs != 0 {
		t.Fatalf("warm MatchBatch allocates %v per run, want 0", allocs)
	}
}
```

Append to `match_test.go` (external package; uses the corpus when present):

```go
// TestCorpusBatchEquivalence cross-checks MatchBatch against per-line
// MatchInto over the real corpus. Skips without strs.json.
func TestCorpusBatchEquivalence(t *testing.T) {
	if benchMatcher == nil {
		t.Skip("no strs.json")
	}
	s := benchMatcher.NewSession()
	var res drain3.BatchResult
	var argBuf [64]string
	const chunk = 4096
	for start := 0; start < len(benchLines); start += chunk {
		lines := benchLines[start:min(start+chunk, len(benchLines))]
		s.MatchBatch(lines, &res)
		for i, line := range lines {
			id, args, ok := benchMatcher.MatchInto(line, argBuf[:0])
			got := res.Args[res.ArgOff[i]:res.ArgOff[i+1]]
			if (!ok && (res.IDs[i] != 0 || len(got) != 0)) ||
				(ok && (int(res.IDs[i]) != id || len(got) != len(args))) {
				t.Fatalf("line %d: batch=(%d,%d args) perline=(%d,%v,%d args)", start+i, res.IDs[i], len(got), id, ok, len(args))
			}
			for j := range args {
				if got[j] != args[j] {
					t.Fatalf("line %d arg %d: %q vs %q", start+i, j, got[j], args[j])
				}
			}
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -run 'TestMatchBatch' -count=1 .`
Expected: FAIL — `s.MatchBatch undefined`, `BatchResult undefined`.

- [ ] **Step 3: Implement appendArgs in drain.go and rebase extractArgsInto on it**

Replace `extractArgsInto` with:

```go
// appendArgs appends the tokens at c's param positions to dst and
// returns it. The appended strings alias lineTokens' backing line.
func (c *cluster) appendArgs(dst []string, lineTokens []string, paramID uint64) []string {
	limit := min(len(c.tokenIDs), len(lineTokens))
	for i := 0; i < limit; i++ {
		if c.tokenIDs[i] == paramID {
			dst = append(dst, lineTokens[i])
		}
	}
	return dst
}

func (c *cluster) extractArgsInto(lineTokens []string, paramID uint64, dst []string) []string {
	if len(c.tokenIDs) == 0 || len(lineTokens) == 0 || c.paramCount == 0 {
		return nil
	}
	args := dst[:0]
	if cap(args) < min(c.paramCount, min(len(c.tokenIDs), len(lineTokens))) {
		args = make([]string, 0, c.paramCount)
	}
	args = c.appendArgs(args, lineTokens, paramID)
	if len(args) == 0 {
		return nil
	}
	return args
}
```

- [ ] **Step 4: Implement BatchResult and MatchBatch in session.go**

```go
// BatchResult holds MatchBatch output. Reuse it across calls: all
// slices are truncated and refilled, so a warm result allocates
// nothing. Args entries alias the input lines and keep them alive.
type BatchResult struct {
	IDs    []int32  // per line: matched template ID, 0 = miss
	ArgOff []int32  // prefix offsets, len(lines)+1: line i's args are Args[ArgOff[i]:ArgOff[i+1]]
	Args   []string // extracted args, back to back
}

func (r *BatchResult) reset(n int) {
	if cap(r.IDs) < n {
		r.IDs = make([]int32, 0, n)
		r.ArgOff = make([]int32, 0, n+1)
	}
	r.IDs = r.IDs[:0]
	r.ArgOff = append(r.ArgOff[:0], 0)
	r.Args = r.Args[:0]
}

// MatchBatch matches every line and writes struct-of-arrays results
// into dst, allocating a fresh BatchResult when dst is nil. Template
// IDs are dense and far below MaxInt32 in any trainable dictionary.
func (s *Session) MatchBatch(lines []string, dst *BatchResult) *BatchResult {
	if dst == nil {
		dst = &BatchResult{}
	}
	dst.reset(len(lines))
	paramID := s.m.paramID
	for _, line := range lines {
		cluster, tokens := s.findMatch(line)
		if cluster == nil {
			dst.IDs = append(dst.IDs, 0)
		} else {
			dst.IDs = append(dst.IDs, int32(cluster.id))
			dst.Args = cluster.appendArgs(dst.Args, tokens, paramID)
		}
		dst.ArgOff = append(dst.ArgOff, int32(len(dst.Args)))
	}
	return dst
}
```

- [ ] **Step 5: Run tests**

Run: `go test -race -count=1 .`
Expected: PASS, including TestCorpusBatchEquivalence when strs.json is present (run it once with the corpus: it is the strongest equivalence gate in this task).

- [ ] **Step 6: Add the batch benchmark**

Append to `match_test.go`:

```go
func BenchmarkMatchBatchAll(b *testing.B) {
	if benchMatcher == nil {
		b.Skip("no strs.json")
	}
	s := benchMatcher.NewSession()
	var res drain3.BatchResult
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		s.MatchBatch(benchLines, &res)
	}
	b.ReportMetric(float64(len(benchLines)), "lines/op")
}
```

Run: `go test -run '^$' -bench 'BenchmarkMatchBatchAll|BenchmarkMatchIntoAll' -benchtime 2x .`
Expected: MatchBatchAll ≥ as fast as MatchIntoAll, 0 B/op after warmup (first op may allocate the arena).

- [ ] **Step 7: Update README**

If `README.md` documents the Match API, add after that section:

> For block workloads (e.g. column compression) use `Session.MatchBatch`: `s := m.NewSession(); res := s.MatchBatch(lines, &reusable)` returns struct-of-arrays results (`IDs`, and `Args` with `ArgOff` prefix offsets) and allocates nothing once the reusable result is warm. A trained `Matcher` is immutable — share it across goroutines with one `Session` per goroutine.

- [ ] **Step 8: Commit**

```bash
git add session.go session_test.go drain.go match_test.go README.md
git commit -m "feat: MatchBatch with reusable struct-of-arrays BatchResult"
```

---

### Task 3: Bitmap tokenizer split (SWAR kernel, no asm yet)

**Files:**
- Create: `tokenize.go` (moved+split tokenizer), `tokenize_test.go`
- Modify: `drain.go` (delete moved functions, add Matcher.trainBM), `session.go` (Session.spaceBM), `match.go`/`train.go` call sites, `drain_test.go` (remove moved tests)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `func spaceBitmap(content string, bm []uint64)` — fills 1 bit per byte (bit i set iff content[i]==' '), dense 64 bytes/word, all `(len+63)/64` words written, tail bits zero. Tasks 4-5 swap its scan engine.
  - `func spaceBitmapSWAR(content string, bm []uint64)` — pure-Go kernel, compiled on all platforms, oracle for asm.
  - `func tokenizeWhitespaceCount(content string, dst []string, maxTokens int, bm []uint64) ([]string, int)` — same semantics as today plus the caller-provided bitmap scratch (`bm` cap ≥ `(len(content)+63)/64`).
  - `func bitmapWords(maxBytes int) int { return (maxBytes + 63) / 64 }`

- [ ] **Step 1: Move the tokenizer tests and write the bitmap tests**

Create `tokenize_test.go` (package `drain3`): move `tokenizeWhitespaceRef`, `TestTokenizeWhitespaceCount`, and `BenchmarkTokenizeWhitespaceCount` verbatim from `drain_test.go` (delete them there), updating every `tokenizeWhitespaceCount(s, nil, maxTok)` call to `tokenizeWhitespaceCount(s, nil, maxTok, make([]uint64, bitmapWords(len(s))))`. Then add:

```go
// TestSpaceBitmap locks the bitmap kernels against a naive scan.
func TestSpaceBitmap(t *testing.T) {
	naive := func(s string) []uint64 {
		bm := make([]uint64, bitmapWords(len(s)))
		for i := 0; i < len(s); i++ {
			if s[i] == ' ' {
				bm[i>>6] |= 1 << (i & 63)
			}
		}
		return bm
	}
	rng := rand.New(rand.NewSource(11))
	cases := []string{"", " ", "a", strings.Repeat(" ", 200), strings.Repeat("x", 200)}
	for n := 0; n <= 200; n++ { // every length crossing 64B block boundaries
		b := make([]byte, n)
		for i := range b {
			b[i] = [6]byte{' ', 'a', 0x1F, 0x21, 0x00, 0xFF}[rng.Intn(6)]
		}
		cases = append(cases, string(b))
	}
	for _, s := range cases {
		want := naive(s)
		got := make([]uint64, bitmapWords(len(s)))
		for i := range got {
			got[i] = ^uint64(0) // poison: kernels must overwrite every word
		}
		spaceBitmap(s, got)
		if !slices.Equal(got, want) {
			t.Fatalf("spaceBitmap(%q): got %x, want %x", s, got, want)
		}
		gotSWAR := make([]uint64, bitmapWords(len(s)))
		for i := range gotSWAR {
			gotSWAR[i] = ^uint64(0)
		}
		spaceBitmapSWAR(s, gotSWAR)
		if !slices.Equal(gotSWAR, want) {
			t.Fatalf("spaceBitmapSWAR(%q): got %x, want %x", s, gotSWAR, want)
		}
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `go test -run 'TestSpaceBitmap|TestTokenizeWhitespaceCount' -count=1 .`
Expected: FAIL — `spaceBitmap undefined`, `bitmapWords undefined`.

- [ ] **Step 3: Create tokenize.go**

Move `tokenize()` (the ExtraDelimiters splitter) and `hasNumbers` unchanged from drain.go into `tokenize.go`; delete the old `tokenizeWhitespaceCount` from drain.go (with its `encoding/binary`, `math/bits`, `unsafe`, `strings` imports if now unused there). New content:

```go
package drain3

import (
	"encoding/binary"
	"math/bits"
	"strings"
	"unsafe"
)

// bitmapWords returns the space-bitmap length for maxBytes of input.
func bitmapWords(maxBytes int) int { return (maxBytes + 63) / 64 }

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. All (len+63)/64 words
// are overwritten; bits at or beyond len(content) are zero.
// Platform kernels (NEON/AVX2) replace the scan in Tasks 4-5; until
// then it is the pure-Go SWAR kernel on every platform.
func spaceBitmap(content string, bm []uint64) {
	spaceBitmapSWAR(content, bm)
}

// spaceBitmapSWAR is the pure-Go kernel and the oracle for the
// assembly kernels. Per 8 bytes: the carry-free per-byte equality mask
// (see the proof on the mask in git history at tokenizeWhitespaceCount)
// yields an 0x80-pattern; (m>>7)*0x0102040810204080>>56 compacts it to
// 8 dense bits (byte columns each sum distinct bits < 0x100, so the
// multiply cannot carry between byte lanes).
func spaceBitmapSWAR(content string, bm []uint64) {
	const (
		hi     = 0x8080808080808080
		lo7    = 0x7f7f7f7f7f7f7f7f
		spaces = 0x2020202020202020
	)
	n := len(content)
	if n == 0 {
		return
	}
	buf := unsafe.Slice(unsafe.StringData(content), n)
	w := 0
	i := 0
	for ; i+64 <= n; i += 64 {
		var word uint64
		for k := 0; k < 64; k += 8 {
			x := binary.LittleEndian.Uint64(buf[i+k:]) ^ spaces
			m := ^(((x &^ hi) + lo7) | x) & hi
			word |= ((m >> 7) * 0x0102040810204080) >> 56 << k
		}
		bm[w] = word
		w++
	}
	if i < n {
		var word uint64
		for ; i+8 <= n; i += 8 {
			x := binary.LittleEndian.Uint64(buf[i:]) ^ spaces
			m := ^(((x &^ hi) + lo7) | x) & hi
			word |= ((m >> 7) * 0x0102040810204080) >> 56 << (i & 63)
		}
		for ; i < n; i++ {
			if content[i] == ' ' {
				word |= 1 << (i & 63)
			}
		}
		bm[w] = word
	}
}

// tokenizeWhitespaceCount splits on spaces and returns the token count
// in a single pass over the space bitmap. maxTokens limits emission: if
// the count would exceed maxTokens the function returns early with a
// count > maxTokens so the caller can reject. bm is caller scratch with
// capacity >= bitmapWords(len(content)).
func tokenizeWhitespaceCount(content string, dst []string, maxTokens int, bm []uint64) ([]string, int) {
	if content == "" || maxTokens <= 0 {
		return dst[:0], 0
	}
	words := bitmapWords(len(content))
	bm = bm[:words]
	spaceBitmap(content, bm)
	dst = dst[:0]
	start := 0
	count := 1
	for w := 0; w < words; w++ {
		m := bm[w]
		base := w << 6
		for m != 0 {
			j := base + bits.TrailingZeros64(m)
			m &= m - 1
			dst = append(dst, content[start:j])
			start = j + 1
			count++
			if count > maxTokens {
				return dst, count
			}
		}
	}
	return append(dst, content[start:]), count
}
```

- [ ] **Step 4: Wire the bitmap scratch through the two callers**

`session.go`: add `spaceBM []uint64` to Session; in `NewSession` add `spaceBM: make([]uint64, bitmapWords(m.cfg.MaxBytes))`. In `tokenizeMatchLine` change the call to `tokenizeWhitespaceCount(line, s.tok, s.m.cfg.MaxTokens, s.spaceBM)`.

`drain.go`: add `trainBM []uint64` to Matcher; in `newMatcher` add `m.trainBM = make([]uint64, bitmapWords(cfg.MaxBytes))`.

`train.go` `addLogMessage`: change the call to `tokenizeWhitespaceCount(content, m.scratchTok, m.cfg.MaxTokens, m.trainBM)`.

- [ ] **Step 5: Run the full suite and the regression gate**

Run: `go test -race -count=1 .`
Expected: PASS.

Gate (spec step 2 — the scan/emit split must not regress):
```bash
SP=$(mktemp -d) && go test -c -o $SP/drain3.test . && cd $SP && ./drain3.test -test.run '^$' -test.bench 'BenchmarkTokenizeWhitespaceCount|BenchmarkTrainMatch/match_into$|BenchmarkLargeMixed' -test.benchtime 2s
```
Expected: swar_long ≤ ~85 ns/op, swar_short ≤ ~55 ns/op, match_into and LargeMixed within noise of Task 1's numbers. **If the split regresses per-line by more than ~3%**, keep the fused SWAR body as `tokenizeWhitespaceCount` on non-asm platforms behind build tags and use the bitmap path only where asm exists (record the numbers in the commit message either way).

- [ ] **Step 6: Commit**

```bash
git add tokenize.go tokenize_test.go drain.go drain_test.go session.go match.go train.go
git commit -m "refactor: split tokenizer into spaceBitmap scan kernel + emission

One tokenizer path for per-line and batch matching; the SWAR kernel is
the all-platform default and the oracle for the asm kernels."
```

---

### Task 4: NEON kernel (arm64)

**Files:**
- Create: `tokenize_arm64.s`, `tokenize_asm.go` (tag: `arm64`), `tokenize_noasm.go` (tag: `!arm64`)
- Modify: `tokenize.go` (spaceBitmap becomes the per-platform dispatch), `tokenize_test.go`

**Interfaces:**
- Consumes: `spaceBitmapSWAR`, `bitmapWords` from Task 3.
- Produces: `func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)` (asm, arm64) — one uint64 bitmap word per 64 input bytes; `spaceBitmap` on arm64 = asm blocks + SWAR tail.

- [ ] **Step 1: Extend the differential test**

TestSpaceBitmap from Task 3 already exercises every length 0–200 with poisoned output words against the naive oracle — on arm64 it will now cover the asm path automatically. Add one large case to the `cases` slice to stress many blocks:

```go
	cases = append(cases, strings.Repeat("token pair  ", 512)) // 6KB, block-crossing spaces
```

Also add a fuzz target:

```go
func FuzzSpaceBitmap(f *testing.F) {
	f.Add("a b  c")
	f.Add(strings.Repeat(" x", 100))
	f.Fuzz(func(t *testing.T, s string) {
		want := make([]uint64, bitmapWords(len(s)))
		for i := 0; i < len(s); i++ {
			if s[i] == ' ' {
				want[i>>6] |= 1 << (i & 63)
			}
		}
		got := make([]uint64, bitmapWords(len(s)))
		spaceBitmap(s, got)
		if !slices.Equal(got, want) {
			t.Fatalf("spaceBitmap(%q): got %x want %x", s, got, want)
		}
	})
}
```

- [ ] **Step 2: Move the dispatch into per-platform files**

In `tokenize.go`, delete the `spaceBitmap` function (keep its doc comment moved along). Create `tokenize_noasm.go`:

```go
//go:build !arm64

package drain3

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. All (len+63)/64 words
// are overwritten; bits at or beyond len(content) are zero.
func spaceBitmap(content string, bm []uint64) {
	spaceBitmapSWAR(content, bm)
}
```

Create `tokenize_asm.go`:

```go
//go:build arm64

package drain3

import "unsafe"

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. Full 64-byte blocks go
// through the NEON kernel; the tail is finished by the SWAR kernel so
// no load ever crosses the end of the string.
func spaceBitmap(content string, bm []uint64) {
	n := len(content)
	blocks := n >> 6
	if blocks > 0 {
		spaceBitmapBlocks(unsafe.StringData(content), blocks, &bm[0])
	}
	if tail := n & 63; tail != 0 {
		var w [1]uint64
		spaceBitmapSWAR(content[blocks<<6:], w[:])
		bm[blocks] = w[0]
	}
}

//go:noescape
func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)
```

- [ ] **Step 3: Write the NEON kernel**

`tokenize_arm64.s` — per 64 bytes: four 16-byte compares against 0x20, AND with bit weights {1,2,4,8,16,32,64,128}×2, three VADDP folds to a 64-bit lane, store one word:

```asm
#include "textflag.h"

DATA spaceScanWeights<>+0(SB)/8, $0x8040201008040201
DATA spaceScanWeights<>+8(SB)/8, $0x8040201008040201
GLOBL spaceScanWeights<>(SB), RODATA|NOPTR, $16

// func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)
// One uint64 bitmap word per 64 input bytes: bit i set iff byte i == ' '.
TEXT ·spaceBitmapBlocks(SB), NOSPLIT, $0-24
	MOVD	p+0(FP), R0
	MOVD	blocks+8(FP), R1
	MOVD	bm+16(FP), R2
	MOVD	$spaceScanWeights<>(SB), R3
	VLD1	(R3), [V5.B16]
	VMOVI	$0x20, V4.B16
loop:
	VLD1.P	64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VCMEQ	V4.B16, V0.B16, V0.B16
	VCMEQ	V4.B16, V1.B16, V1.B16
	VCMEQ	V4.B16, V2.B16, V2.B16
	VCMEQ	V4.B16, V3.B16, V3.B16
	VAND	V5.B16, V0.B16, V0.B16
	VAND	V5.B16, V1.B16, V1.B16
	VAND	V5.B16, V2.B16, V2.B16
	VAND	V5.B16, V3.B16, V3.B16
	VADDP	V1.B16, V0.B16, V0.B16
	VADDP	V3.B16, V2.B16, V2.B16
	VADDP	V2.B16, V0.B16, V0.B16
	VADDP	V0.B16, V0.B16, V0.B16
	VMOV	V0.D[0], R4
	MOVD.P	R4, 8(R2)
	SUBS	$1, R1, R1
	BNE	loop
	RET
```

Notes for the implementer: this is the simdjson-style to_bitmask fold. If the Go assembler rejects `VMOVI`, add a second 16-byte RODATA constant of 0x20 bytes and `VLD1` it. If the differential test shows lane-order inversion, the fix is the weights constant or the VADDP operand order — the test in Step 1 diagnoses it immediately (a single space at position k shows up at the wrong bit).

- [ ] **Step 4: Run the differential + fuzz tests**

Run: `go test -run 'TestSpaceBitmap|TestTokenize' -count=1 . && go test -fuzz FuzzSpaceBitmap -fuzztime 30s .`
Expected: PASS. Then the full suite: `go test -race -count=1 .` — PASS (race build exercises asm under the race detector's instrumented harness).

- [ ] **Step 5: Benchmark gate (spec step 3)**

Add to `tokenize_test.go`:

```go
func BenchmarkSpaceBitmap(b *testing.B) {
	long := strings.Repeat("longishtoken ", 24)[: 24*13-1]
	bm := make([]uint64, bitmapWords(len(long)))
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(len(long)))
		for b.Loop() {
			spaceBitmap(long, bm)
		}
	})
	b.Run("swar", func(b *testing.B) {
		b.SetBytes(int64(len(long)))
		for b.Loop() {
			spaceBitmapSWAR(long, bm)
		}
	})
}
```

```bash
SP=$(mktemp -d) && go test -c -o $SP/drain3.test . && cd $SP && ./drain3.test -test.run '^$' -test.bench 'BenchmarkSpaceBitmap|BenchmarkTokenizeWhitespaceCount|BenchmarkTrainMatch/match_into$|BenchmarkLargeMixed' -test.benchtime 2s
```
Expected: kernel ≥ 1.5x swar on the scan; match_into / LargeMixed measurably better than Task 3 (target ~5-15% end to end). Also rerun the corpus benches (`BenchmarkMatchBatchAll`) with strs.json present and record ns/line in the commit message. If NEON does NOT beat SWAR end to end, stop and re-evaluate before Task 5 — do not land a kernel that only wins microbenchmarks.

- [ ] **Step 6: Commit**

```bash
git add tokenize.go tokenize_asm.go tokenize_noasm.go tokenize_arm64.s tokenize_test.go
git commit -m "perf: NEON space-scan kernel on arm64

spaceBitmap dispatches full 64-byte blocks to asm and finishes tails
with the SWAR oracle kernel; differential + fuzz tested against it."
```

---

### Task 5: AVX2 kernel (amd64)

**Files:**
- Create: `tokenize_amd64.s`
- Modify: `tokenize_asm.go` (tag becomes `arm64 || amd64`, amd64 runtime dispatch), `tokenize_noasm.go` (tag becomes `!arm64 && !amd64`), `go.mod` (+ `golang.org/x/sys`)

**Interfaces:**
- Consumes: `spaceBitmapSWAR`, `bitmapWords`, the Task 4 dispatch structure.
- Produces: `func spaceBitmapBlocksAVX2(p *byte, blocks int, bm *uint64)` (asm, amd64); amd64 `spaceBitmap` uses it when `cpu.X86.HasAVX2`, else SWAR.

- [ ] **Step 1: Add the dependency and platform files**

Run: `go get golang.org/x/sys@latest`

Change `tokenize_noasm.go` build tag to `//go:build !arm64 && !amd64`.

Split `tokenize_asm.go` per platform instead of sharing one file (the dispatch differs). Keep the arm64 file as-is but rename tag comments; create `tokenize_amd64.go`:

```go
//go:build amd64

package drain3

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var hasAVX2 = cpu.X86.HasAVX2

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. Full 64-byte blocks go
// through the AVX2 kernel when available (SWAR otherwise); the tail is
// finished by the SWAR kernel so no load ever crosses the string end.
func spaceBitmap(content string, bm []uint64) {
	if !hasAVX2 {
		spaceBitmapSWAR(content, bm)
		return
	}
	n := len(content)
	blocks := n >> 6
	if blocks > 0 {
		spaceBitmapBlocksAVX2(unsafe.StringData(content), blocks, &bm[0])
	}
	if tail := n & 63; tail != 0 {
		var w [1]uint64
		spaceBitmapSWAR(content[blocks<<6:], w[:])
		bm[blocks] = w[0]
	}
}

//go:noescape
func spaceBitmapBlocksAVX2(p *byte, blocks int, bm *uint64)
```

(Rename the arm64 dispatch file from `tokenize_asm.go` to `tokenize_arm64.go` with tag `//go:build arm64` so each platform owns one dispatch file.)

- [ ] **Step 2: Write the AVX2 kernel**

`tokenize_amd64.s`:

```asm
#include "textflag.h"

// func spaceBitmapBlocksAVX2(p *byte, blocks int, bm *uint64)
// One uint64 bitmap word per 64 input bytes: bit i set iff byte i == ' '.
TEXT ·spaceBitmapBlocksAVX2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), AX
	MOVQ	blocks+8(FP), CX
	MOVQ	bm+16(FP), DX
	MOVQ	$0x2020202020202020, BX
	MOVQ	BX, X0
	VPBROADCASTQ	X0, Y0
loop:
	VMOVDQU	(AX), Y1
	VMOVDQU	32(AX), Y2
	VPCMPEQB	Y0, Y1, Y1
	VPCMPEQB	Y0, Y2, Y2
	VPMOVMSKB	Y1, BX
	VPMOVMSKB	Y2, SI
	SHLQ	$32, SI
	ORQ	SI, BX
	MOVQ	BX, (DX)
	ADDQ	$64, AX
	ADDQ	$8, DX
	DECQ	CX
	JNZ	loop
	VZEROUPPER
	RET
```

- [ ] **Step 3: Cross-run the differential tests under Rosetta**

Run: `GOARCH=amd64 go test -run 'TestSpaceBitmap|TestTokenize|TestSessionMatchesMatcher|TestMatchBatch' -count=1 .`
Expected: PASS (macOS runs the amd64 binary under Rosetta 2, which supports AVX2 on this OS; correctness only — do not trust Rosetta timings). Also native: `go test -race -count=1 .` — PASS.

- [ ] **Step 4: Commit**

```bash
git add tokenize_amd64.go tokenize_amd64.s tokenize_arm64.go tokenize_noasm.go go.mod go.sum
git commit -m "perf: AVX2 space-scan kernel on amd64 with runtime detection

Falls back to the SWAR kernel when AVX2 is absent; correctness pinned
by the same differential and fuzz tests as the NEON kernel."
```

---

### Task 6: End-to-end verification, panel review, push

**Files:** none new (fixes only if gates fail)

- [ ] **Step 1: Corpus equivalence against the pre-branch commit**

With strs.json present: `go test -run 'TestCorpusBatchEquivalence' -count=1 .` — PASS. This plus the unchanged unit suite is the byte-identity gate (per-line path was pinned continuously by every task).

- [ ] **Step 2: Full benchmark matrix, recorded**

```bash
SP=$(mktemp -d) && go test -c -o $SP/drain3.test . && cd $SP && ./drain3.test -test.run '^$' -test.bench 'BenchmarkTrainMatch|BenchmarkLargeMixed|BenchmarkTokenizeWhitespaceCount|BenchmarkSpaceBitmap' -test.benchtime 2s | tee bench_final.txt
```
Then from the repo (corpus benches): `go test -run '^$' -bench 'BenchmarkMatchAll|BenchmarkMatchIntoAll|BenchmarkFindMatchOnly|BenchmarkMatchBatchAll' -benchtime 2x .`
Record the interesting numbers in the final commit/summary.

- [ ] **Step 3: Judge panel**

Run `/goreview auto` with all six judges scoped to `*.go *.s` (keep strs.json out). Apply any fixes the panel requires; re-run the suite after.

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-review notes

- Spec coverage: Session/API → Tasks 1-2; tokenizer split + kernel dispatch → Task 3; NEON → Task 4; AVX2 + x/sys/cpu → Task 5; gates and panel → each task's bench step + Task 6. The staged-SoA pipeline is explicitly out of scope (spec non-goal); prototype only if Task 2/4 numbers suggest headroom, in the scratch harness, never in-repo without numbers.
- The SWAR compaction multiply is proven carry-free in the Task 3 comment (byte columns sum distinct bits < 0x100).
- Asm mnemonic risk is bounded: both kernels are gated by a poisoned-output differential test over every length 0-200 plus fuzz; the plan tells the implementer exactly which two knobs (weights constant, VADDP order) fix a lane-order failure.
- Type consistency: `spaceBitmapBlocks(p *byte, blocks int, bm *uint64)` (arm64) vs `spaceBitmapBlocksAVX2` (amd64) — distinct names, each declared only in its own platform file; `tokenizeWhitespaceCount(content, dst, maxTokens, bm)` signature is identical at both call sites (session, train).
