package compress

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"slices"

	"github.com/axiomhq/drain3"
	"github.com/bits-and-blooms/bitset"
	"github.com/mhr3/streamvbyte"
)

// DrainCompressed provides random access to compressed lines and serialization.
type DrainCompressed interface {
	ValueAt(i int) string
	WriteTo(w io.Writer) (int64, error)
	Len() int
}

// Decompress reconstructs all original lines from a DrainCompressed.
func Decompress(dc DrainCompressed) []string {
	if c, ok := dc.(*compressed); ok {
		return c.decompressAll()
	}
	n := dc.Len()
	lines := make([]string, n)
	for i := range lines {
		lines[i] = dc.ValueAt(i)
	}
	return lines
}

// Compress trains a drain3 matcher on a stride-sampled 10% of lines.
func Compress(lines []string) (DrainCompressed, error) {
	return CompressWithConfig(lines, drain3.DefaultConfig())
}

// CompressWithConfig is like Compress but with a custom drain3 config.
func CompressWithConfig(lines []string, cfg drain3.Config) (DrainCompressed, error) {
	sample := strideSample(lines, 0.10, 64)
	matcher, err := drain3.TrainWithConfig(sample, cfg)
	if err != nil {
		return nil, err
	}
	return CompressWithMatcher(lines, matcher)
}

func strideSample(lines []string, frac float64, blockSize int) []string {
	total := len(lines)
	sampleN := int(float64(total) * frac)
	if sampleN <= 0 || total == 0 {
		return lines
	}
	numBlocks := max(sampleN/blockSize, 1)
	stride := total / numBlocks
	if stride < blockSize {
		stride = blockSize
	}
	out := make([]string, 0, sampleN)
	for start := 0; start < total && len(out) < sampleN; start += stride {
		out = append(out, lines[start:min(start+blockSize, total)]...)
	}
	return out
}

// CompressWithMatcher compresses lines using a pre-trained matcher.
// Templates matching fewer than 2 lines are demoted to unmatched.
// Original drain3 template IDs are remapped to dense indices 0..N-1.
func CompressWithMatcher(lines []string, matcher *drain3.Matcher) (DrainCompressed, error) {
	templates := matcher.Templates()
	sparseByID := make(map[int]drain3.Template, len(templates))
	for _, t := range templates {
		sparseByID[t.ID] = t
	}

	bitmap := bitset.New(uint(len(lines)))
	sparseIDs := make([]int, 0, len(lines))
	rawArgs := make([][]string, 0, len(lines))
	var unmatched []string
	var buf [32]string
	for i, line := range lines {
		templateID, args, ok := matcher.MatchInto(line, buf[:0])
		if !ok {
			unmatched = append(unmatched, line)
			continue
		}
		bitmap.Set(uint(i))
		sparseIDs = append(sparseIDs, templateID)
		rawArgs = append(rawArgs, append([]string(nil), args...))
	}

	sparseMatchCount := make(map[int]int, len(sparseByID))
	for _, id := range sparseIDs {
		sparseMatchCount[id]++
	}

	newBitmap := bitset.New(uint(len(lines)))
	var keptSparseIDs []int
	var keptRawArgs [][]string
	var newUnmatched []string
	matchIdx := 0
	unmatchIdx := 0
	for i := range len(lines) {
		if bitmap.Test(uint(i)) {
			id := sparseIDs[matchIdx]
			if sparseMatchCount[id] >= 2 {
				newBitmap.Set(uint(i))
				keptSparseIDs = append(keptSparseIDs, id)
				keptRawArgs = append(keptRawArgs, rawArgs[matchIdx])
			} else {
				newUnmatched = append(newUnmatched, lines[i])
			}
			matchIdx++
		} else {
			newUnmatched = append(newUnmatched, unmatched[unmatchIdx])
			unmatchIdx++
		}
	}

	uniqueSparse := make(map[int]bool)
	for _, id := range keptSparseIDs {
		uniqueSparse[id] = true
	}
	sortedSparse := make([]int, 0, len(uniqueSparse))
	for id := range uniqueSparse {
		sortedSparse = append(sortedSparse, id)
	}
	slices.Sort(sortedSparse)

	sparseToIndex := make(map[int]uint16, len(sortedSparse))
	tmplSlice := make([]drain3.Template, len(sortedSparse))
	for i, sid := range sortedSparse {
		sparseToIndex[sid] = uint16(i)
		tmplSlice[i] = sparseByID[sid]
	}

	templateIDs := make([]uint16, len(keptSparseIDs))
	for i, sid := range keptSparseIDs {
		templateIDs[i] = sparseToIndex[sid]
	}

	c := &compressed{
		templates:   tmplSlice,
		lineCount:   uint32(len(lines)),
		bitmap:      newBitmap,
		templateIDs: templateIDs,
		unmatched:   newUnmatched,
	}
	c.buildColumnArgs(keptRawArgs)
	return c, nil
}

type templateArgs struct {
	dicts   [][]string
	indices [][]uint32
}

type compressed struct {
	// --- serialized ---
	lineCount   uint32
	bitmap      *bitset.BitSet
	templateIDs []uint16
	templates   []drain3.Template
	argCols     []*templateArgs
	unmatched   []string

	// --- NOT serialized ---
	rowIdx   []uint32
	reconBuf []byte
	argBuf   []string
}

func (c *compressed) buildColumnArgs(rawArgs [][]string) {
	c.rowIdx = make([]uint32, len(c.templateIDs))
	tmplRowCount := make([]uint32, len(c.templates))
	for matchIdx, id := range c.templateIDs {
		c.rowIdx[matchIdx] = tmplRowCount[id]
		tmplRowCount[id]++
	}

	type dictBuilder struct {
		lookup map[string]uint32
		values []string
	}

	c.argCols = make([]*templateArgs, len(c.templates))
	builders := make([][]dictBuilder, len(c.templates))

	for id, rowCount := range tmplRowCount {
		nCols := int(c.templates[id].Params.Count())
		if nCols == 0 || rowCount == 0 {
			continue
		}
		ta := &templateArgs{
			dicts:   make([][]string, nCols),
			indices: make([][]uint32, nCols),
		}
		for col := range nCols {
			ta.indices[col] = make([]uint32, rowCount)
		}
		c.argCols[id] = ta
		bs := make([]dictBuilder, nCols)
		for i := range bs {
			bs[i].lookup = make(map[string]uint32)
		}
		builders[id] = bs
	}

	for matchIdx, id := range c.templateIDs {
		bs := builders[id]
		if bs == nil {
			continue
		}
		row := c.rowIdx[matchIdx]
		ta := c.argCols[id]
		for col, val := range rawArgs[matchIdx] {
			idx, ok := bs[col].lookup[val]
			if !ok {
				idx = uint32(len(bs[col].values))
				bs[col].lookup[val] = idx
				bs[col].values = append(bs[col].values, val)
			}
			ta.indices[col][row] = idx
		}
	}

	for id, bs := range builders {
		if bs == nil {
			continue
		}
		ta := c.argCols[id]
		for col := range bs {
			ta.dicts[col] = bs[col].values
		}
	}
}

func (c *compressed) Len() int { return int(c.lineCount) }

func (c *compressed) gatherArgs(id uint16, row uint32) []string {
	ta := c.argCols[id]
	if ta == nil {
		return c.argBuf[:0]
	}
	c.argBuf = c.argBuf[:0]
	for col := range ta.dicts {
		c.argBuf = append(c.argBuf, ta.dicts[col][ta.indices[col][row]])
	}
	return c.argBuf
}

func (c *compressed) ValueAt(i int) string {
	if i < 0 || i >= int(c.lineCount) {
		panic(fmt.Sprintf("compress: index %d out of range [0, %d)", i, c.lineCount))
	}
	ui := uint(i)
	if c.bitmap.Test(ui) {
		matchIdx := int(c.bitmap.Rank(ui)) - 1
		id := c.templateIDs[matchIdx]
		return c.reconstruct(c.templates[id], c.gatherArgs(id, c.rowIdx[matchIdx]))
	}
	return c.unmatched[i-int(c.bitmap.Rank(ui))]
}

func (c *compressed) decompressAll() []string {
	lines := make([]string, c.lineCount)
	matchIdx := 0
	unmatchIdx := 0
	for i := range c.lineCount {
		if c.bitmap.Test(uint(i)) {
			id := c.templateIDs[matchIdx]
			lines[i] = c.reconstruct(c.templates[id], c.gatherArgs(id, c.rowIdx[matchIdx]))
			matchIdx++
		} else {
			lines[i] = c.unmatched[unmatchIdx]
			unmatchIdx++
		}
	}
	return lines
}

const compressMagic = "dc01"

// stickyWriter provides writes with sticky errors and byte counting.
type stickyWriter struct {
	bw      *bufio.Writer
	n       int64
	err     error
	scratch [8]byte
}

func (sw *stickyWriter) write(p []byte) {
	if sw.err != nil {
		return
	}
	n, err := sw.bw.Write(p)
	sw.n += int64(n)
	sw.err = err
}

func (sw *stickyWriter) writeString(s string) {
	if sw.err != nil {
		return
	}
	n, err := sw.bw.WriteString(s)
	sw.n += int64(n)
	sw.err = err
}

func (sw *stickyWriter) u32(v uint32) {
	binary.LittleEndian.PutUint32(sw.scratch[:4], v)
	sw.write(sw.scratch[:4])
}

func (sw *stickyWriter) u64(v uint64) {
	binary.LittleEndian.PutUint64(sw.scratch[:8], v)
	sw.write(sw.scratch[:8])
}

// writeBlob writes a uint32 length prefix followed by raw bytes.
func (sw *stickyWriter) writeBlob(b []byte) {
	sw.u32(uint32(len(b)))
	sw.write(b)
}

func (sw *stickyWriter) flush() {
	if sw.err != nil {
		return
	}
	sw.err = sw.bw.Flush()
}

// uint16s → uint32s for streamvbyte encoding.
func u16toU32(src []uint16) []uint32 {
	dst := make([]uint32, len(src))
	for i, v := range src {
		dst[i] = uint32(v)
	}
	return dst
}

// svbEncode encodes a uint32 slice with streamvbyte, returns the blob.
func svbEncode(vals []uint32) []byte {
	if len(vals) == 0 {
		return nil
	}
	return streamvbyte.EncodeUint32(vals, nil)
}

// WriteTo serializes using a type-separated layout with streamvbyte-encoded
// integer arrays and raw string bytes, optimized for zstd.
func (c *compressed) WriteTo(w io.Writer) (int64, error) {
	sw := &stickyWriter{bw: bufio.NewWriter(w)}

	sw.write([]byte(compressMagic))
	sw.u32(c.lineCount)

	// === INTEGERS ===

	// Bitmap.
	words := c.bitmap.Words()
	sw.u32(uint32(len(words)))
	for _, word := range words {
		sw.u64(word)
	}

	// TemplateIDs — streamvbyte.
	sw.u32(uint32(len(c.templateIDs)))
	sw.writeBlob(svbEncode(u16toU32(c.templateIDs)))

	// Flat column indices — streamvbyte.
	var flatIdx []uint32
	for _, ta := range c.argCols {
		if ta == nil {
			continue
		}
		for col := range ta.indices {
			flatIdx = append(flatIdx, ta.indices[col]...)
		}
	}
	sw.u32(uint32(len(flatIdx)))
	sw.writeBlob(svbEncode(flatIdx))

	// === TEMPLATES ===

	sw.u32(uint32(len(c.templates)))
	for _, t := range c.templates {
		sw.u32(uint32(t.Count))
		sw.u32(uint32(t.TokenCount))
		pw := t.Params.Words()
		sw.u32(uint32(len(pw)))
		for _, w := range pw {
			sw.u64(w)
		}
		sw.u32(uint32(len(t.Tokens)))
		for _, tok := range t.Tokens {
			sw.u32(uint32(len(tok)))
			sw.writeString(tok)
		}
	}

	// === COLUMN STRUCTURE ===

	for _, ta := range c.argCols {
		if ta == nil {
			sw.u32(0)
			continue
		}
		sw.u32(uint32(len(ta.dicts)))
		for col := range ta.dicts {
			sw.u32(uint32(len(ta.dicts[col])))
			sw.u32(uint32(len(ta.indices[col])))
		}
	}

	// === STRINGS: dict entries (streamvbyte lengths + raw bytes) ===

	var dictLens []uint32
	for _, ta := range c.argCols {
		if ta == nil {
			continue
		}
		for col := range ta.dicts {
			for _, s := range ta.dicts[col] {
				dictLens = append(dictLens, uint32(len(s)))
			}
		}
	}
	sw.u32(uint32(len(dictLens)))
	sw.writeBlob(svbEncode(dictLens))
	for _, ta := range c.argCols {
		if ta == nil {
			continue
		}
		for col := range ta.dicts {
			for _, s := range ta.dicts[col] {
				sw.writeString(s)
			}
		}
	}

	// === STRINGS: unmatched (streamvbyte lengths + raw bytes) ===

	unmLens := make([]uint32, len(c.unmatched))
	for i, s := range c.unmatched {
		unmLens[i] = uint32(len(s))
	}
	sw.u32(uint32(len(unmLens)))
	sw.writeBlob(svbEncode(unmLens))
	for _, s := range c.unmatched {
		sw.writeString(s)
	}

	sw.flush()
	return sw.n, sw.err
}

// ReadFrom deserializes a DrainCompressed.
func ReadFrom(r io.Reader) (DrainCompressed, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if len(data) < 4 || string(data[:4]) != compressMagic {
		return nil, fmt.Errorf("compress: invalid magic")
	}

	rd := &reader{data: data, pos: 4}

	lineCount := rd.u32()

	// Bitmap.
	wordCount := rd.u32()
	words := make([]uint64, wordCount)
	for j := range words {
		words[j] = rd.u64()
	}
	bm := bitset.From(words)

	// TemplateIDs — streamvbyte.
	tmplIDCount := rd.u32()
	tmplIDBlob := rd.blob()
	tmplIDsU32 := svbDecode(tmplIDBlob, int(tmplIDCount))
	templateIDs := make([]uint16, tmplIDCount)
	for i, v := range tmplIDsU32 {
		templateIDs[i] = uint16(v)
	}

	// Flat indices — streamvbyte.
	flatIdxCount := rd.u32()
	flatIdxBlob := rd.blob()
	flatIndices := svbDecode(flatIdxBlob, int(flatIdxCount))

	// Templates.
	tmplCount := rd.u32()
	templates := make([]drain3.Template, tmplCount)
	for i := range templates {
		count := rd.u32()
		tokenCount := rd.u32()
		pwCount := rd.u32()
		pwords := make([]uint64, pwCount)
		for j := range pwords {
			pwords[j] = rd.u64()
		}
		nToks := rd.u32()
		tokens := make([]string, nToks)
		for j := range tokens {
			n := rd.u32()
			tokens[j] = rd.strN(int(n))
		}
		templates[i] = drain3.Template{
			ID: i, Count: int(count), TokenCount: int(tokenCount),
			Tokens: tokens, Params: bitset.From(pwords),
		}
	}

	// Column structure.
	type colMeta struct {
		nCols     int
		dictSizes []uint32
		idxCounts []uint32
	}
	metas := make([]colMeta, tmplCount)
	for i := range metas {
		nCols := rd.u32()
		m := colMeta{nCols: int(nCols), dictSizes: make([]uint32, nCols), idxCounts: make([]uint32, nCols)}
		for col := range nCols {
			m.dictSizes[col] = rd.u32()
			m.idxCounts[col] = rd.u32()
		}
		metas[i] = m
	}

	// Dict strings — streamvbyte lengths + raw bytes.
	dictCount := rd.u32()
	dictLenBlob := rd.blob()
	dictLens := svbDecode(dictLenBlob, int(dictCount))
	dictStrs := make([]string, dictCount)
	for i := range dictStrs {
		dictStrs[i] = rd.strN(int(dictLens[i]))
	}

	// Reconstruct argCols.
	argCols := make([]*templateArgs, tmplCount)
	dictOff := 0
	idxOff := 0
	for i, m := range metas {
		if m.nCols == 0 {
			continue
		}
		ta := &templateArgs{
			dicts:   make([][]string, m.nCols),
			indices: make([][]uint32, m.nCols),
		}
		for col := range m.nCols {
			ta.dicts[col] = dictStrs[dictOff : dictOff+int(m.dictSizes[col])]
			dictOff += int(m.dictSizes[col])
			ta.indices[col] = flatIndices[idxOff : idxOff+int(m.idxCounts[col])]
			idxOff += int(m.idxCounts[col])
		}
		argCols[i] = ta
	}

	// Unmatched — streamvbyte lengths + raw bytes.
	unmCount := rd.u32()
	unmLenBlob := rd.blob()
	unmLens := svbDecode(unmLenBlob, int(unmCount))
	unmatched := make([]string, unmCount)
	for i := range unmatched {
		unmatched[i] = rd.strN(int(unmLens[i]))
	}

	if rd.err != nil {
		return nil, fmt.Errorf("compress: %w", rd.err)
	}
	if rd.pos != len(data) {
		return nil, fmt.Errorf("compress: %d trailing bytes", len(data)-rd.pos)
	}

	if uint32(len(templateIDs)) != uint32(bm.Count()) {
		return nil, fmt.Errorf("compress: templateIDs length %d != bitmap popcount %d", len(templateIDs), bm.Count())
	}

	// Reconstruct rowIdx.
	rowIdx := make([]uint32, len(templateIDs))
	tmplRowCount := make([]uint32, tmplCount)
	for i, id := range templateIDs {
		rowIdx[i] = tmplRowCount[id]
		tmplRowCount[id]++
	}

	return &compressed{
		templates:   templates,
		lineCount:   lineCount,
		bitmap:      bm,
		templateIDs: templateIDs,
		argCols:     argCols,
		rowIdx:      rowIdx,
		unmatched:   unmatched,
	}, nil
}

func (c *compressed) reconstruct(tmpl drain3.Template, args []string) string {
	c.reconBuf = c.reconBuf[:0]
	denseIdx := 0
	argIdx := 0
	for i := range tmpl.TokenCount {
		if i > 0 {
			c.reconBuf = append(c.reconBuf, ' ')
		}
		if tmpl.Params.Test(uint(i)) {
			c.reconBuf = append(c.reconBuf, args[argIdx]...)
			argIdx++
		} else {
			c.reconBuf = append(c.reconBuf, tmpl.Tokens[denseIdx]...)
			denseIdx++
		}
	}
	return string(c.reconBuf)
}

// svbDecode decodes a streamvbyte-encoded blob into count uint32 values.
func svbDecode(blob []byte, count int) []uint32 {
	if count == 0 {
		return nil
	}
	return streamvbyte.DecodeUint32(blob, count, nil)
}

// reader reads fixed-width LE values and blobs from a byte slice.
type reader struct {
	data []byte
	pos  int
	err  error
}

func (r *reader) need(n int) []byte {
	if r.err != nil {
		return nil
	}
	if r.pos+n > len(r.data) {
		r.err = io.ErrUnexpectedEOF
		return nil
	}
	b := r.data[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) u32() uint32 {
	b := r.need(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (r *reader) u64() uint64 {
	b := r.need(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (r *reader) blob() []byte {
	n := r.u32()
	if n == 0 {
		return nil
	}
	return r.need(int(n))
}

func (r *reader) strN(n int) string {
	if n == 0 {
		return ""
	}
	b := r.need(n)
	if b == nil {
		return ""
	}
	return string(b)
}
