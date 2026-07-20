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
