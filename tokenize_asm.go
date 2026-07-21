//go:build arm64 || amd64

package drain3

import "unsafe"

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. All (len+63)/64 words
// are overwritten; bits at or beyond len(content) are zero. Full
// 64-byte blocks go through the platform kernel when available; the
// tail is finished by the SWAR kernel so no load ever crosses the end
// of the string.
func spaceBitmap(content string, bm []uint64) {
	if !haveBlockKernel {
		spaceBitmapSWAR(content, bm)
		return
	}
	n := len(content)
	blocks := n >> 6
	if blocks > 0 {
		spaceBitmapBlocks(unsafe.StringData(content), blocks, &bm[0])
	}
	if tail := n & 63; tail != 0 {
		var w [1]uint64
		spaceBitmapSWAR(content[blocks<<6:], w[:])
		bm[blocks] = w[0]
	}
}
