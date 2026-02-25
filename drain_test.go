package drain3

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"strings"
	"testing"
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

func TestMarshalRoundTrip(t *testing.T) {
	samples := []string{
		"foo 10 bar 20",
		"foo 11 bar 21",
		"alpha beta gamma",
	}
	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	payload, err := m.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	loaded, err := LoadMatcher(payload)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if !reflect.DeepEqual(m.Config(), loaded.Config()) {
		t.Fatalf("config mismatch")
	}
	if !reflect.DeepEqual(m.Templates(), loaded.Templates()) {
		t.Fatalf("templates mismatch")
	}

	queries := []string{"foo 99 bar 100", "alpha beta gamma", "alpha beta"}
	for _, q := range queries {
		id1, args1, ok1 := m.Match(q)
		id2, args2, ok2 := loaded.Match(q)
		if ok1 != ok2 || id1 != id2 || !reflect.DeepEqual(args1, args2) {
			t.Fatalf("round-trip mismatch for %q: (%d,%v,%v) != (%d,%v,%v)", q, id1, args1, ok1, id2, args2, ok2)
		}
	}
}

func TestUnmarshalCorruptPayload(t *testing.T) {
	m, err := Train([]string{"a b c"})
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}
	payload, err := m.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	if _, err := LoadMatcher(payload[:3]); err == nil {
		t.Fatalf("expected error for truncated payload")
	}

	bad := append([]byte(nil), payload...)
	bad[0] = 'X'
	if _, err := LoadMatcher(bad); err == nil {
		t.Fatalf("expected error for invalid magic")
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

	paramString := DefaultConfig().ParamString
	templateParamCounts := make([]int, 0, len(clusters))
	for _, cluster := range clusters {
		templateID := uint32(len(templateParamCounts))
		clusterTemplateIDs[cluster.ID] = templateID
		templateParamCounts = append(templateParamCounts, countParamTokens(cluster.Tokens, paramString))
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

	paramString := m.Config().ParamString
	templateParamCounts := make([]int, len(templates))
	for templateID, template := range templates {
		clusterToTemplate[template.ID] = templateID
		templateParamCounts[templateID] = countParamTokens(template.Tokens, paramString)
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

		rendered, ok := renderTemplateWithParams(templates[templateID].Tokens, params, paramString)
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

func renderTemplateWithParams(templateTokens []string, params []string, paramString string) (string, bool) {
	rendered := make([]string, len(templateTokens))
	paramIdx := 0
	for i, token := range templateTokens {
		if token != paramString {
			rendered[i] = token
			continue
		}
		if paramIdx >= len(params) {
			return "", false
		}
		rendered[i] = params[paramIdx]
		paramIdx++
	}
	if paramIdx != len(params) {
		return "", false
	}
	return strings.Join(rendered, " "), true
}

func countParamTokens(tokens []string, paramString string) int {
	count := 0
	for _, token := range tokens {
		if token == paramString {
			count++
		}
	}
	return count
}

func BenchmarkDataJSONTrain10PercentThenMatchAll(b *testing.B) {
	const (
		path       = "data.json"
		sampleRate = 0.10
		seed       = 42
	)

	if _, err := os.Stat(path); err != nil {
		b.Skipf("benchmark corpus not found at %q: %v", path, err)
	}

	rng := rand.New(rand.NewSource(seed))
	trainRows, totalRows, err := sampleJSONLines(path, sampleRate, rng)
	if err != nil {
		b.Fatalf("sample train rows: %v", err)
	}
	if totalRows == 0 {
		b.Fatalf("empty corpus in %q", path)
	}

	b.Run("train_10pct", func(b *testing.B) {
		b.ReportAllocs()
		b.ReportMetric(float64(totalRows), "rows/op")
		b.ReportMetric(float64(len(trainRows)), "train_rows/op")
		for b.Loop() {
			if _, err := Train(trainRows); err != nil {
				b.Fatalf("train matcher: %v", err)
			}
		}
	})

	m, err := Train(trainRows)
	if err != nil {
		b.Fatalf("train matcher: %v", err)
	}

	b.Run("match_all_with_pretrained_matcher", func(b *testing.B) {
		b.ReportAllocs()
		b.ReportMetric(float64(totalRows), "rows/op")
		b.ReportMetric(float64(len(trainRows)), "train_rows/op")
		for b.Loop() {
			var matchedRows int
			err := forEachJSONLine(path, func(line string) error {
				_, ok := m.MatchID(line)
				if ok {
					matchedRows++
				}
				return nil
			})
			if err != nil {
				b.Fatalf("match rows: %v", err)
			}
			b.ReportMetric(float64(matchedRows), "matched_rows/op")
		}
	})
}

func sampleJSONLines(path string, sampleRate float64, rng *rand.Rand) ([]string, int, error) {
	if sampleRate <= 0 || sampleRate > 1 {
		return nil, 0, fmt.Errorf("sample rate must be in (0,1], got %f", sampleRate)
	}
	var (
		sample    []string
		totalRows int
		firstLine string
	)
	err := forEachJSONLine(path, func(line string) error {
		totalRows++
		if firstLine == "" {
			firstLine = line
		}
		if rng.Float64() < sampleRate {
			sample = append(sample, line)
		}
		return nil
	})
	if err != nil {
		return nil, 0, err
	}
	if totalRows > 0 && len(sample) == 0 {
		sample = append(sample, firstLine)
	}
	return sample, totalRows, nil
}

func forEachJSONLine(path string, fn func(string) error) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open data file: %w", err)
	}
	defer f.Close()

	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	tok, err := dec.Token()
	if err != nil {
		return fmt.Errorf("read json opening token: %w", err)
	}
	delim, ok := tok.(json.Delim)
	if !ok || delim != '[' {
		return errors.New("data.json must be a JSON array")
	}

	for dec.More() {
		var line string
		if err := dec.Decode(&line); err != nil {
			return fmt.Errorf("decode line: %w", err)
		}
		if err := fn(line); err != nil {
			return err
		}
	}

	tok, err = dec.Token()
	if err != nil {
		return fmt.Errorf("read json closing token: %w", err)
	}
	delim, ok = tok.(json.Delim)
	if !ok || delim != ']' {
		return errors.New("data.json must end with JSON array closing bracket")
	}

	return nil
}
