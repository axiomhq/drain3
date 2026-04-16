package drain3

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

// renderTemplatePlaceholders expands a dense Template into its full
// token sequence, inserting paramStr at each position marked in Params.
func renderTemplatePlaceholders(t Template, paramStr string) string {
	out := make([]string, t.TokenCount)
	denseIdx := 0
	for i := 0; i < t.TokenCount; i++ {
		if t.Params.Test(uint(i)) {
			out[i] = paramStr
		} else {
			out[i] = t.Tokens[denseIdx]
			denseIdx++
		}
	}
	return strings.Join(out, " ")
}

// ---------------------------------------------------------------------
// Reference behaviour: scenarios ported from logpai/Drain3 test_drain.py
// ---------------------------------------------------------------------

// TestLogpaiSSHDScenario replays the canonical 6-line SSHD example with
// sim_th=0.4 and verifies the algorithm collapses it to two templates.
func TestLogpaiSSHDScenario(t *testing.T) {
	samples := []string{
		"Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]",
		"Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]",
		"Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2",
		"Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2",
		"Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2",
		"Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]",
	}

	cfg := DefaultConfig()
	cfg.SimilarityThreshold = 0.4 // logpai default
	m, err := TrainWithConfig(samples, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	want := map[string]int{
		"Dec 10 <*> LabSZ <*> input_userauth_request: invalid user <*> [preauth]":             3,
		"Dec 10 <*> LabSZ <*> Failed password for invalid user <*> from 0.0.0.0 port <*> ssh2": 3,
	}
	got := map[string]int{}
	total := 0
	for _, tmpl := range m.Templates() {
		got[renderTemplatePlaceholders(tmpl, cfg.ParamString)] = tmpl.Count
		total += tmpl.Count
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("templates mismatch\nwant: %#v\ngot:  %#v", want, got)
	}
	if total != len(samples) {
		t.Fatalf("total count: got %d, want %d", total, len(samples))
	}
}

// TestLogpaiSSHDScenarioHighSim replays the sim_th=0.75 variant: the
// higher threshold blocks the 10-token input_userauth lines from
// merging, so four distinct clusters survive.
func TestLogpaiSSHDScenarioHighSim(t *testing.T) {
	samples := []string{
		"Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]",
		"Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]",
		"Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2",
		"Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2",
		"Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2",
		"Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]",
	}

	cfg := DefaultConfig()
	cfg.SimilarityThreshold = 0.75
	m, err := TrainWithConfig(samples, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	want := map[string]int{
		samples[0]: 1,
		samples[1]: 1,
		"Dec 10 <*> LabSZ <*> Failed password for invalid user <*> from 0.0.0.0 port <*> ssh2": 3,
		samples[5]: 1,
	}
	got := map[string]int{}
	total := 0
	for _, tmpl := range m.Templates() {
		got[renderTemplatePlaceholders(tmpl, cfg.ParamString)] = tmpl.Count
		total += tmpl.Count
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("templates mismatch\nwant: %#v\ngot:  %#v", want, got)
	}
	if total != len(samples) {
		t.Fatalf("total count: got %d, want %d", total, len(samples))
	}
}

// TestLogpaiShortMessage exercises inputs shorter than the tree depth:
// a repeated short message reuses its cluster while a distinct one
// creates a new cluster at the same token-count leaf.
func TestLogpaiShortMessage(t *testing.T) {
	m, err := Train([]string{"hello", "hello", "otherword"})
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	got := map[string]int{}
	for _, tmpl := range m.Templates() {
		got[renderTemplatePlaceholders(tmpl, "<*>")] = tmpl.Count
	}
	want := map[string]int{"hello": 2, "otherword": 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("templates mismatch\nwant: %#v\ngot:  %#v", want, got)
	}
}

// TestLogpaiMatchOnly covers Match()'s perfect-similarity semantics:
// wildcard positions absorb any token, literal positions must match
// exactly, and unknown shapes return no match.
func TestLogpaiMatchOnly(t *testing.T) {
	// After training these four lines:
	//   #1: "aa aa <*>" (size 3)
	//   #2: "xx yy zz"  (size 1)
	m, err := Train([]string{"aa aa aa", "aa aa bb", "aa aa cc", "xx yy zz"})
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	cases := []struct {
		line string
		want int // expected cluster id, or 0 for "no match"
	}{
		{"aa aa tt", 1}, // wildcard absorbs tt
		{"xx yy zz", 2}, // exact
		{"xx yy rr", 0}, // literal mismatch
		{"nothing", 0},  // unknown token count
	}
	for _, tc := range cases {
		id, _, ok := m.Match(tc.line)
		switch {
		case tc.want == 0 && ok:
			t.Errorf("Match(%q): got id=%d, want no match", tc.line, id)
		case tc.want != 0 && !ok:
			t.Errorf("Match(%q): got no match, want id=%d", tc.line, tc.want)
		case tc.want != 0 && id != tc.want:
			t.Errorf("Match(%q): got id=%d, want id=%d", tc.line, id, tc.want)
		}
	}
}

// ---------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------

// TestDeterministicTemplates asserts Train is pure: the same input
// produces byte-identical templates every run.
func TestDeterministicTemplates(t *testing.T) {
	samples := []string{
		"svc 1 INFO user 10",
		"svc 2 INFO user 20",
		"svc 3 ERROR user 30",
		"svc 4 ERROR user 40",
	}
	m1, err := Train(samples)
	if err != nil {
		t.Fatalf("train m1: %v", err)
	}
	m2, err := Train(samples)
	if err != nil {
		t.Fatalf("train m2: %v", err)
	}
	if !reflect.DeepEqual(m1.Templates(), m2.Templates()) {
		t.Fatalf("templates are not deterministic\n%v\n%v", m1.Templates(), m2.Templates())
	}
}

// TestTrainHandlesEmptyInput confirms an empty corpus trains without
// error and matches nothing.
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

// TestZeroThresholdsAreValid pins down the behavior at the lower
// threshold bound: 0.0 is accepted and maximally permissive.
func TestZeroThresholdsAreValid(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SimilarityThreshold = 0.0
	cfg.MatchThreshold = 0.0
	m, err := TrainWithConfig([]string{"A B C", "A B D"}, cfg)
	if err != nil {
		t.Fatalf("expected 0.0 thresholds to be valid, got error: %v", err)
	}
	// MatchThreshold=0.0 accepts any tree-routable candidate.
	if _, _, ok := m.Match("A X Y"); !ok {
		t.Fatalf("expected match with 0.0 match threshold")
	}
	// SimilarityThreshold=0.0 merges aggressively: one template.
	if n := len(m.Templates()); n != 1 {
		t.Fatalf("expected 1 template with 0.0 similarity, got %d", n)
	}
}

// TestMaxClusters verifies the cap is honored and that removing it
// yields more clusters on the same input.
func TestMaxClusters(t *testing.T) {
	// Each line has a unique first token, so uncapped each gets its own cluster.
	lines := []string{"alpha X Y", "bravo X Y", "charlie X Y", "delta X Y", "echo X Y"}

	cfg := DefaultConfig()
	cfg.MaxClusters = 2
	capped, err := TrainWithConfig(lines, cfg)
	if err != nil {
		t.Fatalf("train capped: %v", err)
	}
	if n := len(capped.Templates()); n > 2 {
		t.Fatalf("expected at most 2 templates, got %d", n)
	}

	cfg.MaxClusters = 0
	full, err := TrainWithConfig(lines, cfg)
	if err != nil {
		t.Fatalf("train uncapped: %v", err)
	}
	if len(full.Templates()) <= len(capped.Templates()) {
		t.Fatalf("expected uncapped training to produce more templates: uncapped=%d capped=%d",
			len(full.Templates()), len(capped.Templates()))
	}
}

// ---------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------

func TestTrainWithConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Depth = 2
	if _, err := TrainWithConfig([]string{"a b c"}, cfg); err == nil {
		t.Fatalf("expected error for invalid depth")
	}
}

func TestZeroValueConfigIsRejected(t *testing.T) {
	if _, err := TrainWithConfig([]string{"a b c"}, Config{}); err == nil {
		t.Fatalf("expected error for zero-value Config{}")
	}
}

// ---------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------

// TestExtraDelimiters checks that ExtraDelimiters are replaced with
// spaces before tokenization, so "=" splits "k=v" into "k" and "v".
func TestExtraDelimiters(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ExtraDelimiters = []string{"="}
	m, err := TrainWithConfig([]string{"k=v a=1", "k=v a=2"}, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
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

// TestMatchInto verifies MatchInto matches the Match result and writes
// extracted args into the caller-provided buffer (no allocation).
func TestMatchInto(t *testing.T) {
	samples := []string{
		"service 1 level INFO user 10 action 5",
		"service 2 level INFO user 20 action 5",
		"service 3 level INFO user 30 action 5",
	}
	m, err := Train(samples)
	if err != nil {
		t.Fatalf("train: %v", err)
	}

	line := "service 99 level INFO user 777 action 5"
	idA, argsA, okA := m.Match(line)

	scratch := make([]string, 1, 8)
	idB, argsB, okB := m.MatchInto(line, scratch[:0])
	if idA != idB || okA != okB || !reflect.DeepEqual(argsA, argsB) {
		t.Fatalf("MatchInto mismatch: Match=(%d,%v,%v) MatchInto=(%d,%v,%v)",
			idA, argsA, okA, idB, argsB, okB)
	}
	if len(argsB) == 0 {
		t.Fatalf("expected extracted params")
	}
	if &argsB[0] != &scratch[0] {
		t.Fatalf("expected MatchInto to reuse destination buffer")
	}

	_, argsMiss, okMiss := m.MatchInto("short unmatched", scratch[:0])
	if okMiss {
		t.Fatalf("expected no match")
	}
	if len(argsMiss) != 0 {
		t.Fatalf("expected empty args on miss, got %v", argsMiss)
	}
}

// TestConfigAndTemplatesAreCopied guards the public API against
// accidental aliasing: mutations to returned Config / Templates must
// not affect the Matcher's internal state.
func TestConfigAndTemplatesAreCopied(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ExtraDelimiters = []string{"="}
	m, err := TrainWithConfig([]string{"k=v a=1", "k=v a=2"}, cfg)
	if err != nil {
		t.Fatalf("train: %v", err)
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

// ---------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------

// BenchmarkTrainMatch covers the two training regimes and the two
// matching regimes that matter for regression tracking:
//
//   - train_merge: cyclic workload with 5 repeating shapes — exercises
//     the tree-search + merge-into-existing-cluster hot path.
//   - train_fanout: many distinct, digit-free first tokens — exercises
//     cluster creation up to MaxChildren and wildcard-child promotion
//     once the fanout saturates. (Not all 5000 lines create clusters;
//     once the wildcard child exists, later lines merge through it.)
//   - match_into: steady-state hit path with full arg extraction into
//     a reused scratch buffer.
//   - match_miss: steady-state reject path — exercises the prefilter /
//     anchor / unknown-first-token short-circuits.
func BenchmarkTrainMatch(b *testing.B) {
	const nLines = 5000
	rng := rand.New(rand.NewSource(99))

	// Merge workload: 5 repeating shapes, lots of cluster reuse.
	prefixes := []string{
		"svc auth user %d status ok latency %d region us-east-1 method GET",
		"svc billing user %d amount %d currency USD region eu-west-1 method POST",
		"svc gateway request %d upstream %d duration 42 region ap-south-1 method PUT",
		"svc storage bucket %d object %d size 1024 region us-west-2 method DELETE",
		"svc scheduler job %d worker %d priority high region us-central-1 method PATCH",
	}
	mergeLines := make([]string, nLines)
	for i := range mergeLines {
		mergeLines[i] = fmt.Sprintf(prefixes[i%len(prefixes)], rng.Intn(10000), rng.Intn(10000))
	}

	// Fanout workload: unique, digit-free first tokens so each line
	// exercises the tree-descent + child-creation path until MaxChildren
	// saturates, after which lines route through the wildcard child.
	fanoutLines := make([]string, nLines)
	for i := range fanoutLines {
		a := byte('a' + (i/676)%26)
		b := byte('a' + (i/26)%26)
		c := byte('a' + i%26)
		fanoutLines[i] = fmt.Sprintf("host%c%c%c svc request id %d status ok", a, b, c, rng.Intn(10000))
	}

	trainMerge := mergeLines[:nLines/10]
	matchLines := mergeLines

	b.Run("train_merge", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := Train(trainMerge); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(trainMerge)), "lines/op")
	})

	b.Run("train_fanout", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := Train(fanoutLines); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(fanoutLines)), "lines/op")
	})

	m, err := Train(trainMerge)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("templates: %d", len(m.Templates()))

	b.Run("match_into", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range matchLines {
				if _, _, ok := m.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(matchLines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})

	// Big-dict workload: many token-count buckets × ~MaxChildren
	// first-token variants × diverse per-cluster tokens, so training
	// pushes many distinct strings through the dict. Goal: stress the
	// first-token dict lookup that dominates the reject path.
	bigLines := make([]string, 30000)
	for i := range bigLines {
		tc := 6 + i%8 // 6..13 tokens, spreads across 8 buckets
		// ~780 unique 3-letter first tokens means ~MaxChildren per bucket.
		a := byte('a' + (i/676)%26)
		bc := byte('a' + (i/26)%26)
		cc := byte('a' + i%26)
		host := string([]byte{'h', 'o', 's', 't', a, bc, cc})
		var sb strings.Builder
		sb.WriteString(host)
		for t := 1; t < tc; t++ {
			sb.WriteByte(' ')
			switch t % 4 {
			case 0:
				sb.WriteString(fmt.Sprintf("req-%d", rng.Intn(10000)))
			case 1:
				sb.WriteString("status")
			case 2:
				sb.WriteString("ok")
			case 3:
				sb.WriteString(fmt.Sprintf("code-%d", rng.Intn(1000)))
			}
		}
		bigLines[i] = sb.String()
	}
	mBig, err := Train(bigLines)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("bigdict templates: %d, dict size: %d", len(mBig.Templates()), len(mBig.dictIDs))

	b.Run("match_bigdict_hit", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range bigLines {
				if _, _, ok := mBig.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(bigLines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})

	bigMiss := make([]string, len(bigLines))
	for i, l := range bigLines {
		bigMiss[i] = "zzzzz-unknown " + l
	}
	b.Run("match_bigdict_miss", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range bigMiss {
				if _, _, ok := mBig.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(bigMiss)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})

	missLines := make([]string, len(matchLines))
	for i, l := range matchLines {
		missLines[i] = "zzz-unknown " + l
	}
	b.Run("match_miss", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range missLines {
				if _, _, ok := m.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(missLines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})
}

// BenchmarkLargeMixed models a realistic production workload:
// 2.1M lines where ~90% conform to trained templates and ~10% are
// never-seen-before garbage. Training uses a random 10% sample of
// the full corpus; matching runs against the full 2.1M.
func BenchmarkLargeMixed(b *testing.B) {
	const total = 2_100_000
	rng := rand.New(rand.NewSource(123))

	structuredN := total * 9 / 10 // 1.89M matching-shape lines
	garbageN := total - structuredN

	// Build structured + garbage in separate pools, then interleave so
	// the match workload sees a realistic mix but training only sees
	// structured lines.
	structured := make([]string, structuredN)
	for i := range structured {
		a := byte('a' + (i/676)%26)
		b2 := byte('a' + (i/26)%26)
		c := byte('a' + i%26)
		host := string([]byte{'h', a, b2, c})
		switch i % 5 {
		case 0:
			structured[i] = fmt.Sprintf("%s svc auth user %d status ok latency %d region us-east-1 method GET",
				host, rng.Intn(100000), rng.Intn(10000))
		case 1:
			structured[i] = fmt.Sprintf("%s svc billing user %d amount %d currency USD region eu-west-1 method POST",
				host, rng.Intn(100000), rng.Intn(1000000))
		case 2:
			structured[i] = fmt.Sprintf("%s svc gateway request %d upstream %d duration 42 region ap-south-1 method PUT",
				host, rng.Intn(100000), rng.Intn(10000))
		case 3:
			structured[i] = fmt.Sprintf("%s svc storage bucket %d object %d size 1024 region us-west-2 method DELETE",
				host, rng.Intn(100000), rng.Intn(10000))
		case 4:
			structured[i] = fmt.Sprintf("%s svc scheduler job %d worker %d priority high region us-central-1 method PATCH",
				host, rng.Intn(100000), rng.Intn(10000))
		}
	}
	garbage := make([]string, garbageN)
	for i := range garbage {
		garbage[i] = fmt.Sprintf("zzgarbage-%d random payload %d tail %d", i, rng.Intn(1<<30), rng.Intn(1000))
	}

	// Interleave: every 10th slot is garbage.
	lines := make([]string, total)
	gi, si := 0, 0
	for i := range lines {
		if i%10 == 0 && gi < garbageN {
			lines[i] = garbage[gi]
			gi++
		} else {
			lines[i] = structured[si]
			si++
		}
	}

	// Train on 10% of the structured pool only — the model must not
	// have seen any garbage during training.
	idx := rng.Perm(structuredN)[:total/10]
	trainLines := make([]string, len(idx))
	for i, j := range idx {
		trainLines[i] = structured[j]
	}
	m, err := Train(trainLines)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("trained: templates=%d, dict=%d, train_lines=%d, total=%d",
		len(m.Templates()), len(m.dictIDs), len(trainLines), total)

	b.Run("match_all", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]string, 0, 16)
		var matched int
		for b.Loop() {
			matched = 0
			for _, line := range lines {
				if _, _, ok := m.MatchInto(line, scratch[:0]); ok {
					matched++
				}
			}
		}
		b.ReportMetric(float64(len(lines)), "lines/op")
		b.ReportMetric(float64(matched), "matched/op")
	})
}
