package drain3

import (
	"encoding/binary"
	"math/bits"
	"strings"
	"unsafe"
)

// bitmapWords returns the space-bitmap length for maxBytes of input.
func bitmapWords(maxBytes int) int { return (maxBytes + 63) / 64 }

// spaceBitmap (defined per-platform in tokenize_asm.go / tokenize_noasm.go)
// fills bm with one bit per input byte: bit i is set iff content[i] == ' ',
// packed 64 bytes per word. All (len+63)/64 words are overwritten; bits at
// or beyond len(content) are zero.

// spaceBitmapSWAR is the pure-Go kernel and the oracle for the
// assembly kernels. Per 8 bytes: the carry-free per-byte equality mask
// (see the proof on the mask in git history at tokenizeWhitespaceCount)
// yields an 0x80-pattern; (m>>7)*0x0102040810204080>>56 compacts it to
// 8 dense bits (byte columns each sum distinct bits < 0x100, so the
// multiply cannot carry between byte lanes).
func spaceBitmapSWAR(content string, bm []uint64) {
	const (
		hi     = 0x8080808080808080
		lo7    = 0x7f7f7f7f7f7f7f7f
		spaces = 0x2020202020202020
	)
	n := len(content)
	if n == 0 {
		return
	}
	buf := unsafe.Slice(unsafe.StringData(content), n)
	w := 0
	i := 0
	for ; i+64 <= n; i += 64 {
		var word uint64
		for k := 0; k < 64; k += 8 {
			x := binary.LittleEndian.Uint64(buf[i+k:]) ^ spaces
			m := ^(((x &^ hi) + lo7) | x) & hi
			word |= ((m >> 7) * 0x0102040810204080) >> 56 << k
		}
		bm[w] = word
		w++
	}
	if i < n {
		var word uint64
		for ; i+8 <= n; i += 8 {
			x := binary.LittleEndian.Uint64(buf[i:]) ^ spaces
			m := ^(((x &^ hi) + lo7) | x) & hi
			word |= ((m >> 7) * 0x0102040810204080) >> 56 << (i & 63)
		}
		for ; i < n; i++ {
			if content[i] == ' ' {
				word |= 1 << (i & 63)
			}
		}
		bm[w] = word
	}
}

// tokenizeWhitespaceCount splits on spaces and returns the token count
// in a single pass over the space bitmap. maxTokens limits emission: if
// the count would exceed maxTokens the function returns early with a
// count > maxTokens so the caller can reject. bm is caller scratch with
// capacity >= bitmapWords(len(content)).
func tokenizeWhitespaceCount(content string, dst []string, maxTokens int, bm []uint64) ([]string, int) {
	if content == "" || maxTokens <= 0 {
		return dst[:0], 0
	}
	words := bitmapWords(len(content))
	bm = bm[:words]
	spaceBitmap(content, bm)
	dst = dst[:0]
	start := 0
	count := 1
	for w := 0; w < words; w++ {
		m := bm[w]
		base := w << 6
		for m != 0 {
			j := base + bits.TrailingZeros64(m)
			m &= m - 1
			dst = append(dst, content[start:j])
			start = j + 1
			count++
			if count > maxTokens {
				return dst, count
			}
		}
	}
	return append(dst, content[start:]), count
}

func tokenize(content string, extraDelimiters []string) []string {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}
	for _, delimiter := range extraDelimiters {
		content = strings.ReplaceAll(content, delimiter, " ")
	}
	return strings.Fields(content)
}

func hasNumbers(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			return true
		}
	}
	return false
}
