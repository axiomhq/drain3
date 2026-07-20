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

// TestSessionRebindsAfterRebuild pins the default session's back-pointer
// after NewMatcherFromTemplates, which publishes the matcher by value
// (*m = *next) and would otherwise leave the session pointing at the
// discarded build-time Matcher.
func TestSessionRebindsAfterRebuild(t *testing.T) {
	src, err := Train([]string{
		"svc auth user 1 ip 10.0.0.1",
		"svc auth user 2 ip 10.0.0.2",
	})
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	m, err := NewMatcherFromTemplates(src.Config(), src.Templates())
	if err != nil {
		t.Fatalf("rebuild: %v", err)
	}
	if id, ok := m.MatchID("svc auth user 9 ip 1.2.3.4"); !ok || id == 0 {
		t.Fatalf("rebuilt matcher: got (%d,%v), want a hit", id, ok)
	}
	if m.defaultSession == nil || m.defaultSession.m != m {
		t.Fatalf("default session back-pointer not rebound to the published Matcher")
	}
}

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
