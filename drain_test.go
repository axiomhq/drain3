package drain3

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/bits-and-blooms/bitset"
)

func TestTrainMatchBasic(t *testing.T) {
	samples := []string{
		"A B 100",
		"A B 200",
		"X Y Z",
	}
	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	id, args, ok := m.Match("A B 999")
	if !ok {
		t.Fatalf("expected match")
	}
	if id != 1 {
		t.Fatalf("expected template 1, got %d", id)
	}
	if !reflect.DeepEqual(args, []string{"999"}) {
		t.Fatalf("unexpected args: %v", args)
	}

	id, args, ok = m.Match("X Y Z")
	if !ok || id != 2 {
		t.Fatalf("expected exact match for template 2, got id=%d ok=%v", id, ok)
	}
	if args != nil {
		t.Fatalf("expected nil args, got %v", args)
	}

	_, _, ok = m.Match("X Y Q")
	if ok {
		t.Fatalf("expected no match")
	}
}

func TestDeterministicTemplates(t *testing.T) {
	samples := []string{
		"svc 1 INFO user 10",
		"svc 2 INFO user 20",
		"svc 3 ERROR user 30",
		"svc 4 ERROR user 40",
	}
	m1, err := Train(samples)
	if err != nil {
		t.Fatalf("train m1 failed: %v", err)
	}
	m2, err := Train(samples)
	if err != nil {
		t.Fatalf("train m2 failed: %v", err)
	}

	if !reflect.DeepEqual(m1.Templates(), m2.Templates()) {
		t.Fatalf("templates are not deterministic\n%v\n%v", m1.Templates(), m2.Templates())
	}
}

func TestConfigAndTemplatesAreCopied(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ExtraDelimiters = []string{"="}
	m, err := TrainWithConfig([]string{"k=v a=1", "k=v a=2"}, cfg)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	readCfg := m.Config()
	readCfg.ExtraDelimiters[0] = ":"
	if m.Config().ExtraDelimiters[0] != "=" {
		t.Fatalf("config getter leaked mutable slice")
	}

	templates := m.Templates()
	templates[0].Tokens[0] = "mutated"
	if reflect.DeepEqual(templates, m.Templates()) {
		t.Fatalf("templates getter leaked mutable data")
	}
}

func TestTrainWithConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Depth = 2
	if _, err := TrainWithConfig([]string{"a b c"}, cfg); err == nil {
		t.Fatalf("expected error for invalid depth")
	}
}

func TestZeroThresholdsAreValid(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SimilarityThreshold = 0.0
	cfg.MatchThreshold = 0.0
	m, err := TrainWithConfig([]string{"A B C", "A B D"}, cfg)
	if err != nil {
		t.Fatalf("expected 0.0 thresholds to be valid, got error: %v", err)
	}
	// With MatchThreshold 0.0, required score = 0, so any candidate
	// reachable via the prefix tree is accepted regardless of similarity.
	// "A X Y" routes through "A" at the tree root then scores 0 — accepted.
	if _, _, ok := m.Match("A X Y"); !ok {
		t.Fatalf("expected match with 0.0 match threshold for routable line")
	}

	// With SimilarityThreshold 0.0, training merges aggressively.
	// "A B C" and "A B D" should produce one template: "A B <*>".
	templates := m.Templates()
	if len(templates) != 1 {
		t.Fatalf("expected 1 template with 0.0 similarity, got %d", len(templates))
	}
}

func TestMaxClusters(t *testing.T) {
	// Each line has a unique first token, so without MaxClusters each gets its own cluster.
	lines := []string{
		"alpha X Y",
		"bravo X Y",
		"charlie X Y",
		"delta X Y",
		"echo X Y",
	}

	cfg := DefaultConfig()
	cfg.MaxClusters = 2
	m, err := TrainWithConfig(lines, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	templates := m.Templates()
	if len(templates) > 2 {
		t.Fatalf("expected at most 2 templates, got %d", len(templates))
	}

	// Without cap, all lines should produce more clusters.
	cfg.MaxClusters = 0
	mFull, err := TrainWithConfig(lines, cfg)
	if err != nil {
		t.Fatalf("train uncapped: %v", err)
	}
	if len(mFull.Templates()) <= len(templates) {
		t.Fatalf("expected uncapped training to produce more templates: got %d vs capped %d",
			len(mFull.Templates()), len(templates))
	}
}

func TestMaxClustersSerializationRoundTrip(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxClusters = 50
	m, err := TrainWithConfig([]string{"A B C", "A B D"}, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	payload, err := m.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	loaded, err := LoadMatcher(payload)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.Config().MaxClusters != 50 {
		t.Fatalf("MaxClusters round-trip: got %d, want 50", loaded.Config().MaxClusters)
	}
}

func TestZeroValueConfigIsRejected(t *testing.T) {
	if _, err := TrainWithConfig([]string{"a b c"}, Config{}); err == nil {
		t.Fatalf("expected error for zero-value Config{}")
	}
}

func TestTrainHandlesEmptyInput(t *testing.T) {
	m, err := Train(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(m.Templates()); got != 0 {
		t.Fatalf("expected no templates, got %d", got)
	}

	if _, _, ok := m.Match("anything"); ok {
		t.Fatalf("expected no match")
	}
}

func TestTrainWithConfigImmediateMatch(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ExtraDelimiters = []string{"="}
	m, err := TrainWithConfig([]string{"k=v a=1", "k=v a=2"}, cfg)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	id, args, ok := m.Match("k=v a=7")
	if !ok {
		t.Fatalf("expected match")
	}
	if id != 1 {
		t.Fatalf("expected template id 1, got %d", id)
	}
	if len(args) != 1 || args[0] != "7" {
		t.Fatalf("unexpected args: %v", args)
	}
}

func TestMatchIDParity(t *testing.T) {
	samples := []string{
		"service 1 level INFO user 10 action 5",
		"service 2 level INFO user 20 action 5",
		"service 9 level WARN user 30 action 8",
	}
	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	cases := []string{
		"service 77 level INFO user 999 action 5",
		"service 9 level WARN user 33 action 8",
		"service 1 level ERROR user 10 action 5",
	}
	for _, line := range cases {
		idWithArgs, _, okWithArgs := m.Match(line)
		idOnly, okOnly := m.MatchID(line)
		if idOnly != idWithArgs || okOnly != okWithArgs {
			t.Fatalf("mismatch for %q: Match=(id=%d ok=%v) MatchID=(id=%d ok=%v)", line, idWithArgs, okWithArgs, idOnly, okOnly)
		}
	}
}

func TestMatchIntoParityAndReuse(t *testing.T) {
	samples := []string{
		"service 1 level INFO user 10 action 5",
		"service 2 level INFO user 20 action 5",
		"service 3 level INFO user 30 action 5",
	}
	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	line := "service 99 level INFO user 777 action 5"
	idA, argsA, okA := m.Match(line)

	scratch := make([]string, 1, 8)
	idB, argsB, okB := m.MatchInto(line, scratch[:0])
	if idA != idB || okA != okB || !reflect.DeepEqual(argsA, argsB) {
		t.Fatalf("MatchInto mismatch: Match=(%d,%v,%v) MatchInto=(%d,%v,%v)", idA, argsA, okA, idB, argsB, okB)
	}
	if len(argsB) == 0 {
		t.Fatalf("expected extracted params")
	}
	if &argsB[0] != &scratch[0] {
		t.Fatalf("expected MatchInto to reuse destination buffer")
	}

	_, argsNoMatch, okNoMatch := m.MatchInto("short unmatched", scratch[:0])
	if okNoMatch {
		t.Fatalf("expected no match")
	}
	if len(argsNoMatch) != 0 {
		t.Fatalf("expected empty args on miss, got %v", argsNoMatch)
	}
}

func TestTokenizeWhitespaceIntoPreservesSpaceBoundaries(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    []string
		wantNil bool
	}{
		{name: "single space", in: "a b", want: []string{"a", "b"}},
		{name: "double spaces", in: "a  b", want: []string{"a", "", "b"}},
		{name: "triple spaces", in: "a   b", want: []string{"a", "", "", "b"}},
		{name: "leading space", in: " a", want: []string{"", "a"}},
		{name: "trailing space", in: "a ", want: []string{"a", ""}},
		{name: "only spaces", in: "  ", want: []string{"", "", ""}},
		{name: "unicode tokens", in: "α  β γ", want: []string{"α", "", "β", "γ"}},
		{name: "empty string", in: "", wantNil: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := tokenizeWhitespaceInto(tc.in, nil)
			if tc.wantNil {
				if got != nil {
					t.Fatalf("expected nil, got %v", got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("tokenizeWhitespaceInto(%q) = %v, want %v", tc.in, got, tc.want)
			}

			scratch := make([]string, 0, 8)
			gotReuse := tokenizeWhitespaceInto(tc.in, scratch[:0])
			if !reflect.DeepEqual(gotReuse, tc.want) {
				t.Fatalf("tokenizeWhitespaceInto with scratch (%q) = %v, want %v", tc.in, gotReuse, tc.want)
			}
		})
	}
}

func TestMatchIsSensitiveToRepeatedSpaces(t *testing.T) {
	m, err := Train([]string{"a  b"})
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	if _, _, ok := m.Match("a  b"); !ok {
		t.Fatalf("expected double-space query to match double-space template")
	}
	if _, _, ok := m.Match("a b"); ok {
		t.Fatalf("expected single-space query not to match double-space template")
	}
}

func TestMatchWithWildcardPositions(t *testing.T) {
	samples := []string{
		"a mid c",
		"b mid c",
		"x edge y",
		"x edge z",
		"p core q",
		"r core s",
		"literal line",
	}

	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	// "x edge y" + "x edge z" merge into "x edge <*>" during training.
	// Tree routes on first token, so only lines starting with a known
	// first token can match.
	if _, _, ok := m.Match("x edge t"); !ok {
		t.Fatalf("expected match for 'x edge t' (first token 'x' in tree, pos 2 is param)")
	}
	if _, _, ok := m.Match("literal line"); !ok {
		t.Fatalf("expected match for 'literal line'")
	}
	if _, _, ok := m.Match("literal  line"); ok {
		t.Fatalf("expected no match for 'literal  line' (different token count)")
	}
	if _, _, ok := m.Match("no match line"); ok {
		t.Fatalf("expected no match for unknown first token")
	}
}

func TestColumnStyleClusterRemapFlow(t *testing.T) {
	const invalidTemplateID = ^uint32(0)

	trainingLines := []string{
		"svc auth user 100 status ok",
		"svc auth user 200 status ok",
		"svc billing user 12 status fail",
		"literal no params line",
	}
	m, err := Train(trainingLines)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	clusters := m.Templates()
	if len(clusters) == 0 {
		t.Fatalf("expected templates")
	}

	maxClusterID := 0
	for _, cluster := range clusters {
		if cluster.ID <= 0 {
			t.Fatalf("expected positive cluster ID, got %d", cluster.ID)
		}
		if cluster.ID > maxClusterID {
			maxClusterID = cluster.ID
		}
	}

	clusterTemplateIDs := make([]uint32, maxClusterID+1)
	for i := range clusterTemplateIDs {
		clusterTemplateIDs[i] = invalidTemplateID
	}

	templateParamCounts := make([]int, 0, len(clusters))
	for _, cluster := range clusters {
		templateID := uint32(len(templateParamCounts))
		clusterTemplateIDs[cluster.ID] = templateID
		templateParamCounts = append(templateParamCounts, countParamTokens(cluster.Params))
	}

	scratch := make([]string, 0, 64)
	lines := []string{
		"svc auth user 999 status ok",
		"svc billing user 33 status fail",
		"literal no params line",
	}

	for _, line := range lines {
		clusterID, params, matched := m.MatchInto(line, scratch[:0])
		if !matched {
			t.Fatalf("expected match for %q", line)
		}
		if clusterID <= 0 || clusterID >= len(clusterTemplateIDs) {
			t.Fatalf("cluster ID out of range: %d", clusterID)
		}

		templateID := clusterTemplateIDs[clusterID]
		if templateID == invalidTemplateID {
			t.Fatalf("no mapped template ID for cluster ID %d", clusterID)
		}

		expectedParamCount := templateParamCounts[int(templateID)]
		if len(params) != expectedParamCount {
			t.Fatalf("param count mismatch for line %q: got %d want %d", line, len(params), expectedParamCount)
		}
	}

	_, _, matched := m.MatchInto("no_match", scratch[:0])
	if matched {
		t.Fatalf("expected no match for unmatched line")
	}
}

func TestLosslessRoundTripWithRawFallback(t *testing.T) {
	const invalidTemplateID = -1

	trainingLines := []string{
		"svc auth user 100 status ok",
		"svc auth user 200 status ok",
		"alpha  beta",
		"alpha  gamma",
		"literal no params line",
	}
	m, err := Train(trainingLines)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	templates := m.Templates()
	if len(templates) == 0 {
		t.Fatalf("expected templates")
	}

	maxClusterID := 0
	for _, template := range templates {
		if template.ID > maxClusterID {
			maxClusterID = template.ID
		}
	}
	clusterToTemplate := make([]int, maxClusterID+1)
	for i := range clusterToTemplate {
		clusterToTemplate[i] = invalidTemplateID
	}

	templateParamCounts := make([]int, len(templates))
	for templateID, template := range templates {
		clusterToTemplate[template.ID] = templateID
		templateParamCounts[templateID] = countParamTokens(template.Params)
	}

	lines := []string{
		"svc auth user 300 status ok",   // matched template + params
		"svc auth user 300 status fail", // unmatched -> raw fallback
		"alpha  delta",                  // matched template + spaces preserved
		"alpha delta",                   // unmatched -> raw fallback
		"literal no params line",        // matched exact template
	}

	out := make([]string, len(lines))
	scratch := make([]string, 0, 16)
	matchedCount := 0
	rawFallbackCount := 0
	for i, line := range lines {
		clusterID, params, matched := m.MatchInto(line, scratch[:0])
		if !matched {
			out[i] = line
			rawFallbackCount++
			continue
		}

		if clusterID <= 0 || clusterID >= len(clusterToTemplate) {
			t.Fatalf("cluster ID out of range: %d", clusterID)
		}
		templateID := clusterToTemplate[clusterID]
		if templateID == invalidTemplateID {
			t.Fatalf("no template mapping for cluster ID %d", clusterID)
		}

		expectedParams := templateParamCounts[templateID]
		if len(params) != expectedParams {
			out[i] = line
			rawFallbackCount++
			continue
		}

		rendered, ok := renderTemplateWithParams(templates[templateID], params)
		if !ok {
			out[i] = line
			rawFallbackCount++
			continue
		}

		out[i] = rendered
		matchedCount++
	}

	if matchedCount == 0 {
		t.Fatalf("expected at least one matched row")
	}
	if rawFallbackCount == 0 {
		t.Fatalf("expected at least one raw fallback row")
	}
	if !reflect.DeepEqual(out, lines) {
		t.Fatalf("lossless round-trip mismatch\nin:  %q\nout: %q", lines, out)
	}
}

func renderTemplateWithParams(tmpl Template, params []string) (string, bool) {
	rendered := make([]string, tmpl.TokenCount)
	denseIdx := 0
	paramIdx := 0
	for i := 0; i < tmpl.TokenCount; i++ {
		if tmpl.Params.Test(uint(i)) {
			if paramIdx >= len(params) {
				return "", false
			}
			rendered[i] = params[paramIdx]
			paramIdx++
		} else {
			rendered[i] = tmpl.Tokens[denseIdx]
			denseIdx++
		}
	}
	if paramIdx != len(params) {
		return "", false
	}
	return strings.Join(rendered, " "), true
}

func countParamTokens(params *bitset.BitSet) int {
	return int(params.Count())
}

// matchEntry holds the compressed representation of a matched line.
type matchEntry struct {
	templateID uint32
	args       []string
}

func BenchmarkCompression(b *testing.B) {
	const (
		path       = "strs.json"
		sampleRate = 0.10
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		b.Skipf("benchmark corpus not found at %q: %v", path, err)
	}

	var allLines []string
	err := forEachJSONLine(path, func(line string) error {
		allLines = append(allLines, line)
		return nil
	})
	if err != nil {
		b.Fatalf("load data: %v", err)
	}
	totalRows := len(allLines)
	if totalRows == 0 {
		b.Fatalf("empty corpus in %q", path)
	}

	// Sample 1% for training.
	rng := rand.New(rand.NewSource(seed))
	var trainRows []string
	for _, line := range allLines {
		if rng.Float64() < sampleRate {
			trainRows = append(trainRows, line)
		}
	}
	if len(trainRows) == 0 {
		trainRows = append(trainRows, allLines[0])
	}

	// Total raw bytes for compression ratio.
	var rawBytes int
	for _, line := range allLines {
		rawBytes += len(line)
	}

	b.Logf("corpus: %d lines, %d train samples (%.1f%%), %.1f MB raw",
		totalRows, len(trainRows), 100*float64(len(trainRows))/float64(totalRows),
		float64(rawBytes)/(1<<20))

	// --- Sub-benchmark: training only ---
	b.Run("train", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := Train(trainRows); err != nil {
				b.Fatalf("train: %v", err)
			}
		}
		b.ReportMetric(float64(len(trainRows)), "train_rows/op")
	})

	// Train once for the matching benchmarks.
	matcher, err := Train(trainRows)
	if err != nil {
		b.Fatalf("train: %v", err)
	}
	templates := matcher.Templates()

	// Build cluster ID → dense template ID remap.
	maxClusterID := 0
	for _, t := range templates {
		if t.ID > maxClusterID {
			maxClusterID = t.ID
		}
	}
	clusterToTemplate := make([]uint32, maxClusterID+1)
	for i := range clusterToTemplate {
		clusterToTemplate[i] = ^uint32(0)
	}
	templateParamCounts := make([]int, len(templates))
	for denseID, t := range templates {
		clusterToTemplate[t.ID] = uint32(denseID)
		templateParamCounts[denseID] = countParamTokens(t.Params)
	}

	b.Logf("templates: %d", len(templates))

	// --- Sub-benchmark: match + build compressed output ---
	b.Run("match_and_compress", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			// Bitmap: bit set = matched, bit clear = unmatched.
			bitmap := make([]uint64, (totalRows+63)/64)
			matches := make([]matchEntry, 0, totalRows)
			unmatched := make([]string, 0, totalRows/10)
			scratch := make([]string, 0, 64)

			for i, line := range allLines {
				clusterID, args, ok := matcher.MatchInto(line, scratch[:0])
				if !ok || clusterID <= 0 || clusterID >= len(clusterToTemplate) || clusterToTemplate[clusterID] == ^uint32(0) {
					unmatched = append(unmatched, line)
					matches = append(matches, matchEntry{}) // placeholder to keep index alignment
					continue
				}
				bitmap[i/64] |= 1 << (uint(i) % 64)
				argsCopy := make([]string, len(args))
				copy(argsCopy, args)
				matches = append(matches, matchEntry{
					templateID: clusterToTemplate[clusterID],
					args:       argsCopy,
				})
			}

			matchedCount := 0
			for _, word := range bitmap {
				matchedCount += popcount(word)
			}

			b.ReportMetric(float64(matchedCount), "matched/op")
			b.ReportMetric(float64(totalRows-matchedCount), "unmatched/op")
			b.ReportMetric(100*float64(matchedCount)/float64(totalRows), "hit_pct/op")

			// Compressed size estimate: bitmap + matched entries + unmatched raw strings.
			compressedBytes := len(bitmap)*8 + // bitmap
				matchedCount*4 // template IDs (uint32)
			for _, m := range matches {
				for _, a := range m.args {
					compressedBytes += len(a)
				}
			}
			for _, s := range unmatched {
				compressedBytes += len(s)
			}
			// Add serialized matcher size.
			payload, _ := matcher.MarshalBinary()
			compressedBytes += len(payload)

			b.ReportMetric(float64(rawBytes)/(1<<20), "raw_MB/op")
			b.ReportMetric(float64(compressedBytes)/(1<<20), "compressed_MB/op")
			if compressedBytes > 0 {
				b.ReportMetric(float64(rawBytes)/float64(compressedBytes), "ratio/op")
			}
			b.ReportMetric(float64(len(payload)), "matcher_bytes/op")
		}
		b.ReportMetric(float64(totalRows), "rows/op")
	})

	// --- Sub-benchmark: match-only (no output allocation) ---
	b.Run("match_only", func(b *testing.B) {
		b.ReportAllocs()
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range allLines {
				if _, ok := matcher.MatchID(line); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(totalRows), "rows/op")
		b.ReportMetric(float64(matched), "matched/op")
	})
}

func popcount(x uint64) int {
	// Kernighan's bit counting.
	n := 0
	for x != 0 {
		x &= x - 1
		n++
	}
	return n
}

func TestTopKCompression(t *testing.T) {
	const (
		path       = "strs.json"
		sampleRate = 0.10
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("corpus not found at %q: %v", path, err)
	}

	var allLines []string
	if err := forEachJSONLine(path, func(line string) error {
		allLines = append(allLines, line)
		return nil
	}); err != nil {
		t.Fatalf("load: %v", err)
	}
	totalRows := len(allLines)

	var rawBytes int
	for _, l := range allLines {
		rawBytes += len(l)
	}

	rng := rand.New(rand.NewSource(seed))
	var sample []string
	for _, l := range allLines {
		if rng.Float64() < sampleRate {
			sample = append(sample, l)
		}
	}

	full, err := Train(sample)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	// Templates() returns sorted by count descending.
	templates := full.Templates()

	for _, topK := range []int{250} {
		topK := topK
		t.Run(fmt.Sprintf("top%d", topK), func(t *testing.T) {
			keep := templates
			if topK < len(keep) {
				keep = keep[:topK]
			}

			// Rebuild matcher from pruned templates.
			m, err := NewMatcherFromTemplates(full.Config(), keep)
			if err != nil {
				t.Fatalf("rebuild: %v", err)
			}

			// Build remap table.
			maxCID := 0
			for _, tmpl := range keep {
				if tmpl.ID > maxCID {
					maxCID = tmpl.ID
				}
			}
			clusterToTemplate := make([]uint32, maxCID+1)
			for i := range clusterToTemplate {
				clusterToTemplate[i] = ^uint32(0)
			}
			for denseID, tmpl := range keep {
				clusterToTemplate[tmpl.ID] = uint32(denseID)
			}

			// Match full corpus.
			scratch := make([]string, 0, 64)
			var matchedCount, unmatchedCount int
			compressedBytes := 0
			for _, line := range allLines {
				cid, args, ok := m.MatchInto(line, scratch[:0])
				if !ok || cid <= 0 || cid >= len(clusterToTemplate) || clusterToTemplate[cid] == ^uint32(0) {
					unmatchedCount++
					compressedBytes += len(line) // raw fallback
					continue
				}
				matchedCount++
				compressedBytes += 4 // template ID
				for _, a := range args {
					compressedBytes += len(a)
				}
			}

			// Bitmap.
			compressedBytes += (totalRows + 63) / 64 * 8

			// Matcher payload.
			payload, _ := m.MarshalBinary()
			compressedBytes += len(payload)

			ratio := float64(rawBytes) / float64(compressedBytes)

			t.Logf("templates: %d | matched: %d (%.1f%%) | unmatched: %d | compressed: %.1f MB | ratio: %.2fx | matcher: %d bytes",
				len(keep), matchedCount, 100*float64(matchedCount)/float64(totalRows),
				unmatchedCount,
				float64(compressedBytes)/(1<<20), ratio,
				len(payload))
		})
	}
}

func TestFullSweep(t *testing.T) {
	const (
		path       = "strs.json"
		sampleRate = 0.10
		matchRate  = 0.10 // subsample corpus for matching to keep runtime reasonable
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("corpus not found at %q: %v", path, err)
	}

	var allLines []string
	if err := forEachJSONLine(path, func(line string) error {
		allLines = append(allLines, line)
		return nil
	}); err != nil {
		t.Fatalf("load: %v", err)
	}

	rng := rand.New(rand.NewSource(seed))
	var sample []string
	for _, l := range allLines {
		if rng.Float64() < sampleRate {
			sample = append(sample, l)
		}
	}

	// Subsample for matching — different seed so it's independent of training sample.
	matchRNG := rand.New(rand.NewSource(seed + 1))
	var matchLines []string
	var matchRawBytes int
	for _, l := range allLines {
		if matchRNG.Float64() < matchRate {
			matchLines = append(matchLines, l)
			matchRawBytes += len(l)
		}
	}
	matchRows := len(matchLines)

	simThs := []float64{0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	topKs := []int{10, 50, 250, 1000}

	type result struct {
		simTh    float64
		topK     int
		hitPct   float64
		savedPct float64
		ratio    float64
	}
	var results []result

	for _, simTh := range simThs {
		cfg := DefaultConfig()
		cfg.SimilarityThreshold = simTh

		full, err := TrainWithConfig(sample, cfg)
		if err != nil {
			t.Fatalf("train: %v", err)
		}

		templates := full.Templates()
		sort.Slice(templates, func(i, j int) bool { return templates[i].Count > templates[j].Count })

		for _, topK := range topKs {
			keep := templates
			if topK < len(keep) {
				keep = keep[:topK]
			}

			m, err := NewMatcherFromTemplates(full.Config(), keep)
			if err != nil {
				t.Fatalf("rebuild: %v", err)
			}

			maxCID := 0
			for _, tmpl := range keep {
				if tmpl.ID > maxCID {
					maxCID = tmpl.ID
				}
			}
			remap := make([]uint32, maxCID+1)
			for i := range remap {
				remap[i] = ^uint32(0)
			}
			for denseID, tmpl := range keep {
				remap[tmpl.ID] = uint32(denseID)
			}

			scratch := make([]string, 0, 64)
			var matchedCount int
			compressedBytes := 0
			for _, line := range matchLines {
				cid, args, ok := m.MatchInto(line, scratch[:0])
				if !ok || cid <= 0 || cid >= len(remap) || remap[cid] == ^uint32(0) {
					compressedBytes += len(line)
					continue
				}
				matchedCount++
				compressedBytes += 4
				for _, a := range args {
					compressedBytes += len(a)
				}
			}
			compressedBytes += (matchRows + 63) / 64 * 8
			payload, _ := m.MarshalBinary()
			compressedBytes += len(payload)

			saved := matchRawBytes - compressedBytes
			results = append(results, result{
				simTh:    simTh,
				topK:     topK,
				hitPct:   100 * float64(matchedCount) / float64(matchRows),
				savedPct: 100 * float64(saved) / float64(matchRawBytes),
				ratio:    float64(matchRawBytes) / float64(compressedBytes),
			})
		}
	}

	sort.Slice(results, func(i, j int) bool { return results[i].ratio > results[j].ratio })

	t.Logf("\n%-5s %-5s | %6s %6s %5s",
		"sim", "topK", "hit%", "saved%", "ratio")
	t.Logf("%s", strings.Repeat("-", 38))
	for _, r := range results {
		t.Logf("%-5.1f %-5d | %5.1f%% %5.1f%% %5.2fx",
			r.simTh, r.topK, r.hitPct, r.savedPct, r.ratio)
	}
}

func TestSimilarityTopKMatrix(t *testing.T) {
	const (
		path       = "strs.json"
		sampleRate = 0.10
		matchRate  = 0.10
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("corpus not found at %q: %v", path, err)
	}

	var allLines []string
	if err := forEachJSONLine(path, func(line string) error {
		allLines = append(allLines, line)
		return nil
	}); err != nil {
		t.Fatalf("load: %v", err)
	}

	rng := rand.New(rand.NewSource(seed))
	var sample []string
	for _, l := range allLines {
		if rng.Float64() < sampleRate {
			sample = append(sample, l)
		}
	}

	matchRNG := rand.New(rand.NewSource(seed + 1))
	var matchLines []string
	var matchRawBytes int
	for _, l := range allLines {
		if matchRNG.Float64() < matchRate {
			matchLines = append(matchLines, l)
			matchRawBytes += len(l)
		}
	}
	matchRows := len(matchLines)

	topKs := []int{10, 50, 250, 1000}
	simThs := []float64{0.3, 0.5, 0.7, 0.9}

	for _, simTh := range simThs {
		cfg := DefaultConfig()
		cfg.SimilarityThreshold = simTh

		full, err := TrainWithConfig(sample, cfg)
		if err != nil {
			t.Fatalf("train sim=%.1f: %v", simTh, err)
		}

		templates := full.Templates()
		sort.Slice(templates, func(i, j int) bool { return templates[i].Count > templates[j].Count })

		for _, topK := range topKs {
			keep := templates
			if topK < len(keep) {
				keep = keep[:topK]
			}

			m, err := NewMatcherFromTemplates(full.Config(), keep)
			if err != nil {
				t.Fatalf("rebuild: %v", err)
			}

			maxCID := 0
			for _, tmpl := range keep {
				if tmpl.ID > maxCID {
					maxCID = tmpl.ID
				}
			}
			clusterToTemplate := make([]uint32, maxCID+1)
			for i := range clusterToTemplate {
				clusterToTemplate[i] = ^uint32(0)
			}
			for denseID, tmpl := range keep {
				clusterToTemplate[tmpl.ID] = uint32(denseID)
			}

			scratch := make([]string, 0, 64)
			var matchedCount int
			compressedBytes := 0
			for _, line := range matchLines {
				cid, args, ok := m.MatchInto(line, scratch[:0])
				if !ok || cid <= 0 || cid >= len(clusterToTemplate) || clusterToTemplate[cid] == ^uint32(0) {
					compressedBytes += len(line)
					continue
				}
				matchedCount++
				compressedBytes += 4
				for _, a := range args {
					compressedBytes += len(a)
				}
			}
			compressedBytes += (matchRows + 63) / 64 * 8
			payload, _ := m.MarshalBinary()
			compressedBytes += len(payload)

			saved := matchRawBytes - compressedBytes
			reduction := 100 * float64(saved) / float64(matchRawBytes)

			t.Logf("sim=%.1f top=%4d | hit=%.1f%% | %.1f MB saved (%.1f%%) | ratio=%.2fx",
				simTh, topK,
				100*float64(matchedCount)/float64(matchRows),
				float64(saved)/(1<<20), reduction,
				float64(matchRawBytes)/float64(compressedBytes))
		}
	}
}

func TestTemplateHitDistribution(t *testing.T) {
	const (
		path       = "strs.json"
		sampleRate = 0.10
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("corpus not found at %q: %v", path, err)
	}

	var allLines []string
	if err := forEachJSONLine(path, func(line string) error {
		allLines = append(allLines, line)
		return nil
	}); err != nil {
		t.Fatalf("load: %v", err)
	}

	rng := rand.New(rand.NewSource(seed))
	var sample []string
	for _, l := range allLines {
		if rng.Float64() < sampleRate {
			sample = append(sample, l)
		}
	}

	m, err := Train(sample)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	// Count hits per template across the full corpus.
	hits := make(map[int]int)
	var missed int
	for _, line := range allLines {
		id, ok := m.MatchID(line)
		if ok {
			hits[id]++
		} else {
			missed++
		}
	}

	type entry struct {
		id    int
		count int
	}
	sorted := make([]entry, 0, len(hits))
	for id, count := range hits {
		sorted = append(sorted, entry{id, count})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].count > sorted[j].count })

	total := len(allLines)
	cumulative := 0
	for i, topK := range []int{10, 25, 50, 100, 250, 500, 1000, len(sorted)} {
		if topK > len(sorted) {
			topK = len(sorted)
		}
		for cumulative == 0 || i > 0 {
			// only recalculate when topK changes
			cumulative = 0
			for j := 0; j < topK; j++ {
				cumulative += sorted[j].count
			}
			break
		}
		// recalculate properly
		cum := 0
		for j := 0; j < topK; j++ {
			cum += sorted[j].count
		}
		t.Logf("top %5d templates: %7d/%d matched (%.1f%% of corpus, %.1f%% of all matches)",
			topK, cum, total,
			100*float64(cum)/float64(total),
			100*float64(cum)/float64(total-missed))
	}
	t.Logf("total templates: %d, missed: %d (%.1f%%)", len(sorted), missed, 100*float64(missed)/float64(total))
}

func BenchmarkTrainMatch(b *testing.B) {
	// Synthetic corpus that exercises tokenizeWhitespaceInto (via Train and Match)
	// and strings.Count (via MatchID/MatchInto fast-reject).
	// Each line has ~10 tokens with a mix of fixed and variable parts.
	const nLines = 5000
	rng := rand.New(rand.NewSource(99))
	prefixes := []string{
		"svc auth user %d status ok latency %d region us-east-1 method GET",
		"svc billing user %d amount %d currency USD region eu-west-1 method POST",
		"svc gateway request %d upstream %d duration 42 region ap-south-1 method PUT",
		"svc storage bucket %d object %d size 1024 region us-west-2 method DELETE",
		"svc scheduler job %d worker %d priority high region us-central-1 method PATCH",
	}

	lines := make([]string, nLines)
	for i := range lines {
		tmpl := prefixes[i%len(prefixes)]
		lines[i] = fmt.Sprintf(tmpl, rng.Intn(10000), rng.Intn(10000))
	}

	// Split: 10% train, rest for matching.
	trainLines := lines[:nLines/10]
	matchLines := lines

	b.Run("train", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := Train(trainLines); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(trainLines)), "lines/op")
	})

	matcher, err := Train(trainLines)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("templates: %d", len(matcher.Templates()))

	b.Run("match_id", func(b *testing.B) {
		b.ReportAllocs()
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range matchLines {
				if _, ok := matcher.MatchID(line); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(matchLines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})

	b.Run("match_into", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range matchLines {
				if _, _, ok := matcher.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(matchLines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})
}

func forEachJSONLine(path string, fn func(string) error) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open data file: %w", err)
	}
	defer f.Close()

	// Fast path: scan for quoted strings directly, avoiding encoding/json overhead.
	// The file is a JSON array of simple strings like: ["line1","line2",...]
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 4<<20), 16<<20)
	scanner.Split(scanJSONStrings)
	for scanner.Scan() {
		line := scanner.Text()
		if err := fn(line); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// scanJSONStrings is a bufio.SplitFunc that extracts unescaped string values
// from a JSON array of strings. It skips delimiters ([, ], ,) and whitespace,
// finds quoted strings, and returns the unescaped content.
func scanJSONStrings(data []byte, atEOF bool) (advance int, token []byte, err error) {
	// Skip to next quote.
	i := 0
	for i < len(data) {
		if data[i] == '"' {
			break
		}
		i++
	}
	if i >= len(data) {
		if atEOF {
			return len(data), nil, nil
		}
		return i, nil, nil
	}

	// Found opening quote at position i. Scan for closing quote.
	start := i + 1
	j := start
	hasEscape := false
	for j < len(data) {
		if data[j] == '\\' {
			hasEscape = true
			j += 2 // skip escaped character
			continue
		}
		if data[j] == '"' {
			// Found closing quote.
			if !hasEscape {
				return j + 1, data[start:j], nil
			}
			// Has escapes - need to unescape.
			unescaped := unescapeJSONString(data[start:j])
			return j + 1, unescaped, nil
		}
		j++
	}

	// Need more data for incomplete string.
	if atEOF {
		return len(data), nil, nil
	}
	return i, nil, nil
}

func unescapeJSONString(s []byte) []byte {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			i++
			switch s[i] {
			case '"', '\\', '/':
				out = append(out, s[i])
			case 'n':
				out = append(out, '\n')
			case 't':
				out = append(out, '\t')
			case 'r':
				out = append(out, '\r')
			default:
				out = append(out, '\\', s[i])
			}
		} else {
			out = append(out, s[i])
		}
	}
	return out
}

func tokenizeManual(content string, dst []string) []string {
	if content == "" {
		return nil
	}
	if dst != nil {
		dst = dst[:0]
	}
	start := 0
	for i := 0; i < len(content); i++ {
		if content[i] == ' ' {
			dst = append(dst, content[start:i])
			start = i + 1
		}
	}
	dst = append(dst, content[start:])
	return dst
}

func tokenizeIndexByte(content string, dst []string) []string {
	if content == "" {
		return nil
	}
	if dst != nil {
		dst = dst[:0]
	}
	for {
		i := strings.IndexByte(content, ' ')
		if i < 0 {
			break
		}
		dst = append(dst, content[:i])
		content = content[i+1:]
	}
	dst = append(dst, content)
	return dst
}

func BenchmarkCrossover(b *testing.B) {
	// Generate strings of various lengths with spaces every ~8 bytes
	for _, size := range []int{32, 64, 128, 256, 512, 1024} {
		var sb strings.Builder
		for sb.Len() < size {
			if sb.Len() > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString("token12")
		}
		line := sb.String()

		b.Run(fmt.Sprintf("manual_%d", size), func(b *testing.B) {
			var buf [128]string
			for b.Loop() {
				tokenizeManual(line, buf[:0])
			}
		})
		b.Run(fmt.Sprintf("indexbyte_%d", size), func(b *testing.B) {
			var buf [128]string
			for b.Loop() {
				tokenizeIndexByte(line, buf[:0])
			}
		})
	}
}
