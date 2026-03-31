package drain3

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

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

func BenchmarkMarshalRoundTrip(b *testing.B) {
	const nLines = 5000
	rng := rand.New(rand.NewSource(42))
	prefixes := []string{
		"svc auth user %d status ok latency %d region us-east-1 method GET",
		"svc billing user %d amount %d currency USD region eu-west-1 method POST",
		"svc gateway request %d upstream %d duration 42 region ap-south-1 method PUT",
		"svc storage bucket %d object %d size 1024 region us-west-2 method DELETE",
	}
	lines := make([]string, nLines)
	for i := range lines {
		lines[i] = fmt.Sprintf(prefixes[i%len(prefixes)], rng.Intn(10000), rng.Intn(10000))
	}

	m, err := Train(lines[:nLines/10])
	if err != nil {
		b.Fatal(err)
	}
	payload, err := m.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("templates: %d, payload: %d bytes", len(m.Templates()), len(payload))

	b.Run("marshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := m.MarshalBinary(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("unmarshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := LoadMatcher(payload); err != nil {
				b.Fatal(err)
			}
		}
	})
}
