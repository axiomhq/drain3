package drain3

import "math/rand"

// strideSample draws roughly frac*len(lines) lines as fixed-size blocks at
// regular strides with random jitter inside each stride window. Uses a seeded
// rng derived from the input length so compression is deterministic — same
// input produces the same serialized bytes across runs and goroutines.
func StrideSample(lines []string, frac float64, blockSize int) []string {
	total := len(lines)
	sampleN := int(float64(total) * frac)
	if sampleN <= 0 || total == 0 {
		return lines
	}
	numBlocks := max(sampleN/blockSize, 1)
	stride := max(total/numBlocks, blockSize)
	rng := rand.New(rand.NewSource(int64(total)))
	out := make([]string, 0, sampleN)
	for start := 0; start < total && len(out) < sampleN; start += stride {
		offset := start + rng.Intn(max(stride-blockSize+1, 1))
		if offset >= total {
			break
		}
		out = append(out, lines[offset:min(offset+blockSize, total)]...)
	}
	return out
}
