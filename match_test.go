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
