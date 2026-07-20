//go:build !arm64

package drain3

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. All (len+63)/64 words
// are overwritten; bits at or beyond len(content) are zero.
func spaceBitmap(content string, bm []uint64) {
	spaceBitmapSWAR(content, bm)
}
