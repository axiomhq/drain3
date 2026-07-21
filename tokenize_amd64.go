//go:build amd64

package drain3

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var hasAVX2 = cpu.X86.HasAVX2

// spaceBitmap fills bm with one bit per input byte: bit i is set iff
// content[i] == ' ', packed 64 bytes per word. Full 64-byte blocks go
// through the AVX2 kernel when available (SWAR otherwise); the tail is
// finished by the SWAR kernel so no load ever crosses the string end.
func spaceBitmap(content string, bm []uint64) {
	if !hasAVX2 {
		spaceBitmapSWAR(content, bm)
		return
	}
	n := len(content)
	blocks := n >> 6
	if blocks > 0 {
		spaceBitmapBlocksAVX2(unsafe.StringData(content), blocks, &bm[0])
	}
	if tail := n & 63; tail != 0 {
		var w [1]uint64
		spaceBitmapSWAR(content[blocks<<6:], w[:])
		bm[blocks] = w[0]
	}
}

//go:noescape
func spaceBitmapBlocksAVX2(p *byte, blocks int, bm *uint64)
