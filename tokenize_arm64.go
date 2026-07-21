//go:build arm64

package drain3

const haveBlockKernel = true

// spaceBitmapBlocks is the NEON kernel, implemented in tokenize_arm64.s.
//
//go:noescape
func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)
