//go:build !arm64 && !amd64

package drain3

// spaceBitmap: see the contract documented on tokenize_asm.go's spaceBitmap.
func spaceBitmap(content string, bm []uint64) {
	spaceBitmapSWAR(content, bm)
}
