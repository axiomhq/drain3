# drain3

A Go implementation of the [Drain log template parser](https://pinjiahe.github.io/papers/ICWS17.pdf).

Drain groups structurally similar log lines into **templates** — token
sequences where positions that vary across lines are replaced with
`<*>`. Typical outcome: millions of lines collapse to hundreds of
templates.

This is a port of [logpai/Drain3](https://github.com/logpai/Drain3),
tuned for Go idioms and read-path performance.

## Install

```sh
go get github.com/axiomhq/drain3
```

## Usage

### Train, then match

```go
import "github.com/axiomhq/drain3"

samples := []string{
    "Dec 10 07:07:38 LabSZ sshd[24206]: invalid user test9 [preauth]",
    "Dec 10 07:08:28 LabSZ sshd[24208]: invalid user webmaster [preauth]",
    "Dec 10 07:28:03 LabSZ sshd[24245]: invalid user pgadmin [preauth]",
}

m, err := drain3.Train(samples)
if err != nil {
    log.Fatal(err)
}

for _, t := range m.Templates() {
    fmt.Printf("id=%d count=%d\n", t.ID, t.Count)
}

id, params, ok := m.Match("Dec 10 09:12:32 LabSZ sshd[24501]: invalid user guest [preauth]")
if ok {
    fmt.Printf("matched template %d with params %v\n", id, params)
}
```

### Custom config

```go
cfg := drain3.DefaultConfig()
cfg.SimilarityThreshold = 0.4          // how aggressively to merge during training
cfg.MatchThreshold      = 1.0          // exact match required at query time (default)
cfg.ExtraDelimiters     = []string{"="}
m, err := drain3.TrainWithConfig(samples, cfg)
```

See `ALGORITHM.md` for what each knob does.

### Zero-allocation matching

`MatchInto` writes extracted args into a caller-provided scratch buffer
so a hot loop can classify without allocating:

```go
scratch := make([]string, 0, 32)
for _, line := range lines {
    id, params, ok := m.MatchInto(line, scratch[:0])
    _ = id
    _ = params
    _ = ok
}
```

### Just the cluster ID

If you don't need the extracted params:

```go
id, ok := m.MatchID(line)
```

## API surface

| Function | Purpose |
| --- | --- |
| `Train(lines) (*Matcher, error)` | Train with default config. |
| `TrainWithConfig(lines, cfg)` | Train with a custom `Config`. |
| `NewMatcherFromTemplates(cfg, templates)` | Rebuild a matcher from previously exported templates. |
| `(*Matcher).Match(line)` | Return `(id, params, ok)`. |
| `(*Matcher).MatchInto(line, buf)` | As `Match`, but writes params into `buf` (zero-alloc on hit). |
| `(*Matcher).MatchID(line)` | Return `(id, ok)` without extracting params. |
| `(*Matcher).Templates()` | Snapshot of trained templates (sorted by count desc). |
| `(*Matcher).Config()` | Copy of the matcher's effective config. |

## Performance

Single-thread on Apple M4, from `BenchmarkLargeMixed` (2.1M lines,
90 % hit / 10 % miss, 474 templates after training on a random 10 %
sample):

| path | ns/line | allocs |
| --- | ---: | ---: |
| match (hit) | ~100 | 0 |
| match (miss) | ~10 | 0 |

Run the full suite:

```sh
go test -bench=. -run=^$ .
```

## References

- Paper: [Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://pinjiahe.github.io/papers/ICWS17.pdf) (ICWS 2017).
- Reference implementation: [logpai/Drain3](https://github.com/logpai/Drain3).
- Algorithm details in this implementation: `ALGORITHM.md`.
