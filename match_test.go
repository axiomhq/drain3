package drain3_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/axiomhq/drain3"
)

var benchLines []string
var benchMatcher *drain3.Matcher

func init() {
	data, err := os.ReadFile("strs.json")
	if err != nil {
		return
	}
	json.Unmarshal(data, &benchLines)
	data = nil

	sampleN := len(benchLines) / 10
	sample := make([]string, 0, sampleN)
	stride := len(benchLines) / (sampleN / 128)
	for i := 0; i < len(benchLines) && len(sample) < sampleN; i += stride {
		end := min(i+128, len(benchLines))
		sample = append(sample, benchLines[i:end]...)
	}

	cfg := drain3.DefaultConfig()
	cfg.SimilarityThreshold = 0.3
	cfg.MaxClusters = 5000
	benchMatcher, _ = drain3.TrainWithConfig(sample, cfg)
}

func BenchmarkMatchAll(b *testing.B) {
	if benchMatcher == nil {
		b.Skip("no strs.json")
	}
	b.ResetTimer()
	for range b.N {
		for _, line := range benchLines {
			benchMatcher.Match(line)
		}
	}
}

func BenchmarkMatchIntoAll(b *testing.B) {
	if benchMatcher == nil {
		b.Skip("no strs.json")
	}
	var buf [32]string
	b.ResetTimer()
	for range b.N {
		for _, line := range benchLines {
			benchMatcher.MatchInto(line, buf[:0])
		}
	}
}

func BenchmarkFindMatchOnly(b *testing.B) {
	if benchMatcher == nil {
		b.Skip("no strs.json")
	}
	b.ResetTimer()
	for range b.N {
		for _, line := range benchLines {
			benchMatcher.MatchID(line)
		}
	}
}

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
