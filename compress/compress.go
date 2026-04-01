package compress

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"slices"

	"github.com/axiomhq/drain3"
	"github.com/bits-and-blooms/bitset"
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

// Compress trains a drain3 matcher on a stride-sampled 10% of lines and
// compresses all lines using the trained matcher.
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

// strideSample takes non-overlapping blocks of blockSize lines, evenly spaced,
// covering approximately frac of the total lines.
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
		end := min(start+blockSize, total)
		out = append(out, lines[start:end]...)
	}
	return out
}

// CompressWithMatcher compresses lines using a pre-trained matcher.
// Templates matching fewer than 2 lines are demoted to unmatched.
func CompressWithMatcher(lines []string, matcher *drain3.Matcher) (DrainCompressed, error) {
	templates := matcher.Templates()
	tmplByID := make(map[int]drain3.Template, len(templates))
	for _, t := range templates {
		tmplByID[t.ID] = t
	}

	// First pass: match all lines.
	bitmap := bitset.New(uint(len(lines)))
	templateIDs := make([]int, 0, len(lines))
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
		templateIDs = append(templateIDs, templateID)
		rawArgs = append(rawArgs, append([]string(nil), args...))
	}

	// Demote templates with < 2 matches to unmatched.
	tmplMatchCount := make(map[int]int, len(tmplByID))
	for _, id := range templateIDs {
		tmplMatchCount[id]++
	}

	newBitmap := bitset.New(uint(len(lines)))
	var newTemplateIDs []int
	var newRawArgs [][]string
	var newUnmatched []string
	matchIdx := 0
	unmatchIdx := 0
	for i := range len(lines) {
		if bitmap.Test(uint(i)) {
			id := templateIDs[matchIdx]
			if tmplMatchCount[id] >= 2 {
				newBitmap.Set(uint(i))
				newTemplateIDs = append(newTemplateIDs, id)
				newRawArgs = append(newRawArgs, rawArgs[matchIdx])
			} else {
				newUnmatched = append(newUnmatched, lines[i])
			}
			matchIdx++
		} else {
			newUnmatched = append(newUnmatched, unmatched[unmatchIdx])
			unmatchIdx++
		}
	}

	// Remove demoted templates.
	activeTmpls := make(map[int]drain3.Template)
	for _, id := range newTemplateIDs {
		if _, ok := activeTmpls[id]; !ok {
			activeTmpls[id] = tmplByID[id]
		}
	}

	c := &compressed{
		tmplByID:    activeTmpls,
		lineCount:   len(lines),
		bitmap:      newBitmap,
		templateIDs: newTemplateIDs,
		unmatched:   newUnmatched,
	}
	c.buildColumnArgs(newRawArgs)
	return c, nil
}

// templateArgs holds dictionary-encoded columnar arg storage for one template.
type templateArgs struct {
	dicts   [][]string // dicts[col] = unique string values for this column
	indices [][]int    // indices[col][row] = index into dicts[col]
}

type compressed struct {
	tmplByID    map[int]drain3.Template
	lineCount   int
	bitmap      *bitset.BitSet
	templateIDs []int
	argCols     map[int]*templateArgs
	rowIdx      []int
	unmatched   []string
	reconBuf    []byte   // reusable; not safe for concurrent use
	argBuf      []string // reusable; not safe for concurrent use
}

// buildColumnArgs converts row-based rawArgs into dictionary-encoded columnar storage.
func (c *compressed) buildColumnArgs(rawArgs [][]string) {
	c.rowIdx = make([]int, len(c.templateIDs))
	tmplRowCount := make(map[int]int, len(c.tmplByID))
	for matchIdx, id := range c.templateIDs {
		c.rowIdx[matchIdx] = tmplRowCount[id]
		tmplRowCount[id]++
	}

	type dictBuilder struct {
		lookup map[string]int
		values []string
	}

	c.argCols = make(map[int]*templateArgs, len(tmplRowCount))
	builders := make(map[int][]dictBuilder, len(tmplRowCount))

	for id, rowCount := range tmplRowCount {
		nCols := int(c.tmplByID[id].Params.Count())
		if nCols == 0 {
			continue
		}
		ta := &templateArgs{
			dicts:   make([][]string, nCols),
			indices: make([][]int, nCols),
		}
		for col := range nCols {
			ta.indices[col] = make([]int, rowCount)
		}
		c.argCols[id] = ta

		bs := make([]dictBuilder, nCols)
		for i := range bs {
			bs[i].lookup = make(map[string]int)
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
				idx = len(bs[col].values)
				bs[col].lookup[val] = idx
				bs[col].values = append(bs[col].values, val)
			}
			ta.indices[col][row] = idx
		}
	}

	for id, bs := range builders {
		ta := c.argCols[id]
		for col := range bs {
			ta.dicts[col] = bs[col].values
		}
	}
}

// Len returns the number of lines.
func (c *compressed) Len() int { return c.lineCount }

func (c *compressed) gatherArgs(id int, row int) []string {
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

// ValueAt reconstructs the string at position i. Panics if out of range.
func (c *compressed) ValueAt(i int) string {
	if i < 0 || i >= c.lineCount {
		panic(fmt.Sprintf("compress: index %d out of range [0, %d)", i, c.lineCount))
	}
	ui := uint(i)
	if c.bitmap.Test(ui) {
		matchIdx := int(c.bitmap.Rank(ui)) - 1
		id := c.templateIDs[matchIdx]
		tmpl := c.tmplByID[id]
		args := c.gatherArgs(id, c.rowIdx[matchIdx])
		return c.reconstruct(tmpl, args)
	}
	unmatchIdx := i - int(c.bitmap.Rank(ui))
	return c.unmatched[unmatchIdx]
}

// decompressAll reconstructs all lines via a linear bitmap walk.
func (c *compressed) decompressAll() []string {
	lines := make([]string, c.lineCount)
	matchIdx := 0
	unmatchIdx := 0
	for i := range c.lineCount {
		if c.bitmap.Test(uint(i)) {
			id := c.templateIDs[matchIdx]
			tmpl := c.tmplByID[id]
			args := c.gatherArgs(id, c.rowIdx[matchIdx])
			lines[i] = c.reconstruct(tmpl, args)
			matchIdx++
		} else {
			lines[i] = c.unmatched[unmatchIdx]
			unmatchIdx++
		}
	}
	return lines
}

const compressMagic = "dc01"

// countWriter wraps a buffered writer with byte counting and sticky errors.
type countWriter struct {
	bw      *bufio.Writer
	n       int64
	err     error
	scratch [binary.MaxVarintLen64]byte
}

func (cw *countWriter) write(p []byte) {
	if cw.err != nil {
		return
	}
	n, err := cw.bw.Write(p)
	cw.n += int64(n)
	cw.err = err
}

func (cw *countWriter) writeString(s string) {
	if cw.err != nil {
		return
	}
	n, err := cw.bw.WriteString(s)
	cw.n += int64(n)
	cw.err = err
}

func (cw *countWriter) flush() {
	if cw.err != nil {
		return
	}
	cw.err = cw.bw.Flush()
}

// WriteTo serializes the compressed representation.
// The output is designed to be maximally compressible by zstd.
func (c *compressed) WriteTo(w io.Writer) (int64, error) {
	cw := &countWriter{bw: bufio.NewWriter(w)}

	// Header.
	copy(cw.scratch[:4], compressMagic)
	cw.write(cw.scratch[:4])
	writeUvarint(cw, uint64(c.lineCount))

	// Bitmap.
	words := c.bitmap.Words()
	writeUvarint(cw, uint64(len(words)))
	for _, word := range words {
		binary.LittleEndian.PutUint64(cw.scratch[:8], word)
		cw.write(cw.scratch[:8])
	}

	// Templates sorted by ID.
	tmplIDs := make([]int, 0, len(c.tmplByID))
	for id := range c.tmplByID {
		tmplIDs = append(tmplIDs, id)
	}
	slices.Sort(tmplIDs)

	writeUvarint(cw, uint64(len(tmplIDs)))
	for _, id := range tmplIDs {
		t := c.tmplByID[id]
		writeVarint(cw, int64(t.ID))
		writeVarint(cw, int64(t.Count))
		writeUvarint(cw, uint64(t.TokenCount))

		paramWords := t.Params.Words()
		writeUvarint(cw, uint64(len(paramWords)))
		for _, pw := range paramWords {
			binary.LittleEndian.PutUint64(cw.scratch[:8], pw)
			cw.write(cw.scratch[:8])
		}

		writeUvarint(cw, uint64(len(t.Tokens)))
		for _, tok := range t.Tokens {
			writeString(cw, tok)
		}
	}

	// TemplateIDs.
	writeUvarint(cw, uint64(len(c.templateIDs)))
	for _, id := range c.templateIDs {
		writeVarint(cw, int64(id))
	}

	// Columnar args.
	var argColIDs []int
	for _, id := range tmplIDs {
		if _, ok := c.argCols[id]; ok {
			argColIDs = append(argColIDs, id)
		}
	}
	writeUvarint(cw, uint64(len(argColIDs)))
	for _, id := range argColIDs {
		ta := c.argCols[id]
		writeVarint(cw, int64(id))
		writeUvarint(cw, uint64(len(ta.dicts)))
		for col := range ta.dicts {
			writeUvarint(cw, uint64(len(ta.dicts[col])))
			for _, s := range ta.dicts[col] {
				writeString(cw, s)
			}
			writeUvarint(cw, uint64(len(ta.indices[col])))
			for _, idx := range ta.indices[col] {
				writeUvarint(cw, uint64(idx))
			}
		}
	}

	// Unmatched.
	writeUvarint(cw, uint64(len(c.unmatched)))
	for _, s := range c.unmatched {
		writeString(cw, s)
	}

	cw.flush()
	return cw.n, cw.err
}

func safeReadUvarint(br *bytes.Reader, label string) (uint64, error) {
	count, err := readUvarint(br)
	if err != nil {
		return 0, fmt.Errorf("compress: read %s: %w", label, err)
	}
	if count > uint64(br.Len()) {
		return 0, fmt.Errorf("compress: %s count %d exceeds remaining bytes %d", label, count, br.Len())
	}
	return count, nil
}

// ReadFrom deserializes a DrainCompressed from r.
func ReadFrom(r io.Reader) (DrainCompressed, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	br := bytes.NewReader(data)

	magic := make([]byte, len(compressMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, fmt.Errorf("compress: read magic: %w", err)
	}
	if string(magic) != compressMagic {
		return nil, fmt.Errorf("compress: invalid magic %q", magic)
	}

	lineCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read line count: %w", err)
	}

	// Bitmap.
	wordCount, err := safeReadUvarint(br, "bitmap word count")
	if err != nil {
		return nil, err
	}
	words := make([]uint64, wordCount)
	for j := range words {
		if err := binary.Read(br, binary.LittleEndian, &words[j]); err != nil {
			return nil, fmt.Errorf("compress: read bitmap word: %w", err)
		}
	}
	bm := bitset.From(words)

	// Templates.
	tmplCount, err := safeReadUvarint(br, "template count")
	if err != nil {
		return nil, err
	}
	tmplByID := make(map[int]drain3.Template, tmplCount)
	for range tmplCount {
		id, err := readVarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read template id: %w", err)
		}
		count, err := readVarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read template count: %w", err)
		}
		tokenCount, err := readUvarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read token count: %w", err)
		}

		pwCount, err := safeReadUvarint(br, "param word count")
		if err != nil {
			return nil, err
		}
		pwords := make([]uint64, pwCount)
		for j := range pwords {
			if err := binary.Read(br, binary.LittleEndian, &pwords[j]); err != nil {
				return nil, fmt.Errorf("compress: read param word: %w", err)
			}
		}

		denseCount, err := safeReadUvarint(br, "dense token count")
		if err != nil {
			return nil, err
		}
		tokens := make([]string, denseCount)
		for j := range tokens {
			tokens[j], err = readStringDirect(br, data)
			if err != nil {
				return nil, fmt.Errorf("compress: read token: %w", err)
			}
		}

		tmplByID[int(id)] = drain3.Template{
			ID:         int(id),
			Tokens:     tokens,
			Params:     bitset.From(pwords),
			TokenCount: int(tokenCount),
			Count:      int(count),
		}
	}

	// TemplateIDs.
	matchCount, err := safeReadUvarint(br, "match count")
	if err != nil {
		return nil, err
	}
	templateIDs := make([]int, matchCount)
	for i := range templateIDs {
		v, err := readVarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read template id: %w", err)
		}
		templateIDs[i] = int(v)
	}

	// Columnar args.
	argColCount, err := safeReadUvarint(br, "arg columns count")
	if err != nil {
		return nil, err
	}
	argCols := make(map[int]*templateArgs, argColCount)
	for range argColCount {
		id, err := readVarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read arg template id: %w", err)
		}
		nCols, err := safeReadUvarint(br, "column count")
		if err != nil {
			return nil, err
		}
		ta := &templateArgs{
			dicts:   make([][]string, nCols),
			indices: make([][]int, nCols),
		}
		for col := range nCols {
			dictSize, err := safeReadUvarint(br, "dict size")
			if err != nil {
				return nil, err
			}
			ta.dicts[col] = make([]string, dictSize)
			for j := range dictSize {
				ta.dicts[col][j], err = readStringDirect(br, data)
				if err != nil {
					return nil, fmt.Errorf("compress: read dict entry: %w", err)
				}
			}
			numIndices, err := safeReadUvarint(br, "indices count")
			if err != nil {
				return nil, err
			}
			ta.indices[col] = make([]int, numIndices)
			for j := range numIndices {
				v, err := readUvarint(br)
				if err != nil {
					return nil, fmt.Errorf("compress: read index: %w", err)
				}
				ta.indices[col][j] = int(v)
			}
		}
		argCols[int(id)] = ta
	}

	// Unmatched.
	unmatchedCount, err := safeReadUvarint(br, "unmatched count")
	if err != nil {
		return nil, err
	}
	unmatched := make([]string, unmatchedCount)
	for i := range unmatched {
		unmatched[i], err = readStringDirect(br, data)
		if err != nil {
			return nil, fmt.Errorf("compress: read unmatched: %w", err)
		}
	}

	if br.Len() != 0 {
		return nil, fmt.Errorf("compress: %d unexpected trailing bytes", br.Len())
	}

	// Validate.
	if len(templateIDs) != int(bm.Count()) {
		return nil, fmt.Errorf("compress: templateIDs length %d does not match bitmap popcount %d", len(templateIDs), bm.Count())
	}
	for i, id := range templateIDs {
		if _, ok := tmplByID[id]; !ok {
			return nil, fmt.Errorf("compress: templateIDs[%d] references unknown template id %d", i, id)
		}
	}
	tmplRowCount := make(map[int]int, len(argCols))
	for _, id := range templateIDs {
		tmplRowCount[id]++
	}
	for id, ta := range argCols {
		expected := tmplRowCount[id]
		for col := range ta.indices {
			if len(ta.indices[col]) != expected {
				return nil, fmt.Errorf("compress: template %d column %d has %d rows, expected %d", id, col, len(ta.indices[col]), expected)
			}
			for _, idx := range ta.indices[col] {
				if idx < 0 || idx >= len(ta.dicts[col]) {
					return nil, fmt.Errorf("compress: template %d column %d index %d out of range [0, %d)", id, col, idx, len(ta.dicts[col]))
				}
			}
		}
	}

	// Reconstruct rowIdx.
	rowIdx := make([]int, len(templateIDs))
	clear(tmplRowCount)
	for i, id := range templateIDs {
		rowIdx[i] = tmplRowCount[id]
		tmplRowCount[id]++
	}

	return &compressed{
		tmplByID:    tmplByID,
		lineCount:   int(lineCount),
		bitmap:      bm,
		templateIDs: templateIDs,
		argCols:     argCols,
		rowIdx:      rowIdx,
		unmatched:   unmatched,
	}, nil
}

func (c *compressed) reconstruct(tmpl drain3.Template, args []string) string {
	paramCount := int(tmpl.Params.Count())
	if len(args) != paramCount {
		panic(fmt.Sprintf("compress: reconstruct: template %d has %d params but got %d args", tmpl.ID, paramCount, len(args)))
	}
	denseCount := tmpl.TokenCount - paramCount
	if len(tmpl.Tokens) != denseCount {
		panic(fmt.Sprintf("compress: reconstruct: template %d expects %d dense tokens but has %d", tmpl.ID, denseCount, len(tmpl.Tokens)))
	}

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

// Binary helpers.

func writeUvarint(cw *countWriter, v uint64) {
	n := binary.PutUvarint(cw.scratch[:], v)
	cw.write(cw.scratch[:n])
}

func writeVarint(cw *countWriter, v int64) {
	n := binary.PutVarint(cw.scratch[:], v)
	cw.write(cw.scratch[:n])
}

func writeString(cw *countWriter, s string) {
	writeUvarint(cw, uint64(len(s)))
	cw.writeString(s)
}

func readUvarint(r *bytes.Reader) (uint64, error) {
	return binary.ReadUvarint(r)
}

func readVarint(r *bytes.Reader) (int64, error) {
	return binary.ReadVarint(r)
}

func readStringDirect(br *bytes.Reader, data []byte) (string, error) {
	n, err := binary.ReadUvarint(br)
	if err != nil {
		return "", err
	}
	if n > uint64(br.Len()) {
		return "", io.ErrUnexpectedEOF
	}
	if n == 0 {
		return "", nil
	}
	start := len(data) - br.Len()
	if _, err := br.Seek(int64(n), io.SeekCurrent); err != nil {
		return "", err
	}
	return string(data[start : start+int(n)]), nil
}
