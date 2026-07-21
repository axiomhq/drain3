package drain3

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
)

// TestTokenizeWhitespaceCount locks the SWAR tokenizer against a naive
// byte-at-a-time reference, including bytes chosen to trigger borrow
// propagation in the SWAR mask (0x1F/0x21 around ' ', 0x01/0x80).
// tokenizeWhitespaceRef is the naive byte-at-a-time tokenizer the SWAR
// implementation replaced: the oracle for TestTokenizeWhitespaceCount and
// the baseline for BenchmarkTokenizeWhitespaceCount.
func tokenizeWhitespaceRef(content string, dst []string, maxTokens int) ([]string, int) {
	if content == "" || maxTokens <= 0 {
		return dst[:0], 0
	}
	dst = dst[:0]
	start, count := 0, 1
	for i := 0; i < len(content); i++ {
		if content[i] != ' ' {
			continue
		}
		dst = append(dst, content[start:i])
		start = i + 1
		count++
		if count > maxTokens {
			return dst, count
		}
	}
	return append(dst, content[start:]), count
}

// TestTokenizeWhitespaceCount locks the SWAR tokenizer against the naive
// reference, including bytes chosen to stress the SWAR mask (0x1F/0x21
// around ' ', 0x01/0x80) and maxTokens truncation.
func TestTokenizeWhitespaceCount(t *testing.T) {
	check := func(s string, maxTok int) {
		t.Helper()
		want, wantN := tokenizeWhitespaceRef(s, nil, maxTok)
		got, gotN := tokenizeWhitespaceCount(s, nil, maxTok, make([]uint64, bitmapWords(len(s))))
		if gotN != wantN || !slices.Equal(got, want) {
			t.Fatalf("tokenize(%q, maxTok=%d): got (%q, %d), want (%q, %d)", s, maxTok, got, gotN, want, wantN)
		}
	}
	edge := []string{
		"", " ", "  ", "a", "a ", " a", "a b", "a  b", "   ",
		"abcdefgh", "abcdefg ", "        x", "x        ",
		"\x01\x80  \x01\x80", "a\x1f b\x21c",
		strings.Repeat(" ", 100), strings.Repeat("ab ", 100),
	}
	for _, s := range edge {
		for _, mt := range []int{0, 1, 2, 3, 64} {
			check(s, mt)
		}
	}
	rng := rand.New(rand.NewSource(7))
	alphabet := []byte{' ', ' ', 0x1F, 0x21, 0xA0, 0x01, 0x80, 'z'}
	for i := 0; i < 5000; i++ {
		b := make([]byte, rng.Intn(40))
		for j := range b {
			b[j] = alphabet[rng.Intn(len(alphabet))]
		}
		check(string(b), 64)
		check(string(b), 3)
	}
}

// naiveSpaceBitmap is the reference oracle for the bitmap kernels.
func naiveSpaceBitmap(s string) []uint64 {
	bm := make([]uint64, bitmapWords(len(s)))
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			bm[i>>6] |= 1 << (i & 63)
		}
	}
	return bm
}

// TestSpaceBitmap locks the bitmap kernels against a naive scan.
func TestSpaceBitmap(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cases := []string{"", " ", "a", strings.Repeat(" ", 200), strings.Repeat("x", 200)}
	cases = append(cases, strings.Repeat("token pair  ", 512)) // 6KB, block-crossing spaces
	for n := 0; n <= 200; n++ {                                // every length crossing 64B block boundaries
		b := make([]byte, n)
		for i := range b {
			b[i] = [6]byte{' ', 'a', 0x1F, 0x21, 0x00, 0xFF}[rng.Intn(6)]
		}
		cases = append(cases, string(b))
	}
	for _, s := range cases {
		want := naiveSpaceBitmap(s)
		got := make([]uint64, bitmapWords(len(s)))
		for i := range got {
			got[i] = ^uint64(0) // poison: kernels must overwrite every word
		}
		spaceBitmap(s, got)
		if !slices.Equal(got, want) {
			t.Fatalf("spaceBitmap(%q): got %x, want %x", s, got, want)
		}
		gotSWAR := make([]uint64, bitmapWords(len(s)))
		for i := range gotSWAR {
			gotSWAR[i] = ^uint64(0)
		}
		spaceBitmapSWAR(s, gotSWAR)
		if !slices.Equal(gotSWAR, want) {
			t.Fatalf("spaceBitmapSWAR(%q): got %x, want %x", s, gotSWAR, want)
		}
	}
}

// BenchmarkTokenizeWhitespaceCount pins the SWAR tokenizer against the
// naive byte-loop reference on two token-length regimes: long tokens
// (~12 B, real-corpus shaped) where the 8-byte windows win most, and
// short tokens (~4 B) where per-window mask reuse must still keep SWAR
// ahead of the byte loop.
func BenchmarkTokenizeWhitespaceCount(b *testing.B) {
	long := strings.Repeat("longishtoken ", 24)[:24*13-1] // 24 tokens, 12 B each
	short := strings.Repeat("svc auth ok 1 ", 6)[:6*14-1] // 24 tokens, ~3 B each
	bm := make([]uint64, bitmapWords(len(long)))
	swar := func(s string, dst []string, maxTokens int) ([]string, int) {
		return tokenizeWhitespaceCount(s, dst, maxTokens, bm)
	}
	for _, tc := range []struct {
		name string
		fn   func(string, []string, int) ([]string, int)
	}{{"swar", swar}, {"ref", tokenizeWhitespaceRef}} {
		for _, in := range []struct {
			name, line string
		}{{"long", long}, {"short", short}} {
			b.Run(tc.name+"_"+in.name, func(b *testing.B) {
				b.SetBytes(int64(len(in.line)))
				dst := make([]string, 0, 64)
				for b.Loop() {
					dst, _ = tc.fn(in.line, dst, 64)
				}
			})
		}
	}
}

// FuzzSpaceBitmap locks spaceBitmap (asm + tail on arm64, SWAR elsewhere)
// against the naive per-byte scan on arbitrary inputs.
func FuzzSpaceBitmap(f *testing.F) {
	f.Add("a b  c")
	f.Add(strings.Repeat(" x", 100))
	f.Fuzz(func(t *testing.T, s string) {
		want := naiveSpaceBitmap(s)
		got := make([]uint64, bitmapWords(len(s)))
		spaceBitmap(s, got)
		if !slices.Equal(got, want) {
			t.Fatalf("spaceBitmap(%q): got %x want %x", s, got, want)
		}
	})
}

// BenchmarkSpaceBitmap compares the platform kernel (NEON blocks + SWAR
// tail on arm64) against the pure-Go SWAR kernel on a real-corpus-shaped
// line.
func BenchmarkSpaceBitmap(b *testing.B) {
	long := strings.Repeat("longishtoken ", 24)[:24*13-1]
	bm := make([]uint64, bitmapWords(len(long)))
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(len(long)))
		for b.Loop() {
			spaceBitmap(long, bm)
		}
	})
	b.Run("swar", func(b *testing.B) {
		b.SetBytes(int64(len(long)))
		for b.Loop() {
			spaceBitmapSWAR(long, bm)
		}
	})
}
