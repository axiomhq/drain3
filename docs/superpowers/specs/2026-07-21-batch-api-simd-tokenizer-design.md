# Batch match API + SIMD tokenizer

Date: 2026-07-21
Status: approved

## Goal

drain3 is the template stage of a column compressor: train on a sample,
match every line, store (template ID, args) instead of raw text. The
per-line match path runs at ~167 ns/line on the reference corpus; the
tokenizer is ~37% of that. This design adds a batch API that a column
codec calls naturally, makes a trained Matcher shareable across
goroutines, and swaps the tokenizer's space scan for platform SIMD
(Go/Plan 9 assembly) with the existing SWAR loop as fallback and oracle.

## Non-goals

- Internal multi-goroutine fan-out inside MatchBatch. One call = one
  core; callers parallelize across blocks with Sessions.
- A staged struct-of-arrays pipeline (tokenize-all → prefilter-all →
  score-all). Hypothesis only; prototyped in the scratch harness after
  step 2 and adopted only if measured end-to-end wins justify it.
- Training-path changes, cgo, GOEXPERIMENT=simd.

## API

```go
// BatchResult holds MatchBatch output. Reuse across calls: slices are
// truncated and refilled, so a warm result allocates nothing.
type BatchResult struct {
    IDs    []int32  // per line: matched template ID, 0 = miss
    ArgOff []int32  // prefix offsets, len(lines)+1: line i's args are Args[ArgOff[i]:ArgOff[i+1]]
    Args   []string // extracted args, back to back
}

func (m *Matcher) NewSession() *Session
func (s *Session) MatchBatch(lines []string, dst *BatchResult) *BatchResult
func (s *Session) Match(line string) (int, []string, bool)
func (s *Session) MatchID(line string) (int, bool)
func (s *Session) MatchInto(line string, dst []string) (int, []string, bool)
func (s *Session) MatchExactInto(line string, dst []string) (int, []string, bool)
```

- Session owns all mutable per-call state (today's scratchTok,
  scratchCandidates, scratchProbeIDs, plus the space-bitmap scratch).
  After training/freeze the Matcher itself is immutable; any number of
  goroutines may share it, one Session per goroutine.
- Existing Matcher.Match* methods keep their signatures and semantics by
  delegating to a lazily created internal Session; the one-goroutine-
  per-Matcher rule continues to apply to them (documented).
- MatchBatch v1 is a tight per-line loop over the existing match path
  writing into the SoA arena. IDs use int32 — template IDs are dense
  and a trainable dictionary cannot approach MaxInt32 clusters — and a
  miss stores 0 and an empty arg span.

## Tokenizer: one path, swappable scan kernel

tokenizeWhitespaceCount splits into scan + emit:

- `spaceBitmap(line string, bm []uint64)` fills a 1-bit-per-byte bitmap
  of space positions (16 words cover the default MaxBytes=1024; each
  Session sizes its bitmap scratch from cfg.MaxBytes at creation).
- Emission walks the bitmap with TrailingZeros64, preserving current
  semantics exactly: split on single ' ', empty tokens kept, count
  returned, maxTokens early exit. The existing oracle test and pinning
  benchmark keep applying to the whole path.

Kernel selection by build tags:

| file | tag | kernel |
|---|---|---|
| tokenize_arm64.s | arm64 | NEON: 4× VLD1+CMEQ($0x20) per 64 B, movemask via bit-weight AND + VADDP folds → uint64 word |
| tokenize_amd64.s | amd64 | AVX2: 2× VPCMPEQB+VPMOVMSKB per 64 B → uint64 word |
| tokenize_generic.go | everything else | today's carry-free SWAR mask, one word per 8 B |

Assembly contract (safety):

- Processes only len&^63 full 64-byte blocks; the Go caller finishes the
  tail with SWAR. No load ever crosses the end of the string (strings
  may end at a page boundary). No writes except the caller's bitmap.
- No token logic, no early exit in asm. MaxBytes bounds over-scan for
  lines that would exceed maxTokens.
- The pure-Go SWAR kernel is compiled on all platforms (exported to
  tests) and serves as the differential oracle for both asm kernels.

## Testing

- Batch/per-line equivalence: MatchBatch vs per-line MatchInto over the
  real corpus and existing synthetic workloads; byte-identical IDs and
  args required.
- Session concurrency: N goroutines, one shared trained Matcher, one
  Session each, run under -race.
- Kernel differential: asm vs SWAR bitmap over all byte values, lengths
  0–130 (crossing block boundaries), random fuzz, and the corpus.
- Benchmarks: MatchBatch vs per-line loop; spaceBitmap kernel pins
  (swar vs asm) alongside the existing tokenizer benchmark.

## Work order and gates

1. Session refactor + MatchBatch (pure Go). Gate: equivalence + -race
   green; per-line path unchanged within noise.
2. Bitmap tokenizer with SWAR kernel. Gate: real-corpus ns/line within
   noise of current (the scan/emit split must not regress).
3. NEON kernel (arm64). Gate: measured end-to-end win on the real
   corpus on M3 Max; expected ~1.5–2x on the scan, ~10–15% end to end.
4. AVX2 kernel (amd64). Correctness gated by the differential test in
   CI; performance validated on a production amd64 host when available.

Each step lands green through the full test suite and the goreview
judge panel before the next begins.
