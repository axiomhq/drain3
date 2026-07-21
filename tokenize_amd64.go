//go:build amd64

package drain3

import "golang.org/x/sys/cpu"

var haveBlockKernel = cpu.X86.HasAVX2

// spaceBitmapBlocks is the AVX2 kernel, implemented in tokenize_amd64.s;
// SWAR is used when AVX2 is absent.
//
//go:noescape
func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)
