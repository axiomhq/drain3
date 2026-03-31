package compress

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"strings"

	"github.com/axiomhq/drain3"
	"github.com/bits-and-blooms/bitset"
)

// DrainCompressed provides random access to compressed lines and serialization.
type DrainCompressed interface {
	ValueAt(i int) string
	WriteTo(w io.Writer) (int64, error)
}

// Compress trains a drain3 matcher on lines and returns a DrainCompressed.
func Compress(lines []string) (DrainCompressed, error) {
	return CompressWithConfig(lines, drain3.DefaultConfig())
}

// CompressWithConfig is like Compress but with a custom drain3 config.
func CompressWithConfig(lines []string, cfg drain3.Config) (DrainCompressed, error) {
	matcher, err := drain3.TrainWithConfig(lines, cfg)
	if err != nil {
		return nil, err
	}
	return CompressWithMatcher(lines, matcher)
}

// CompressWithMatcher compresses lines using a pre-trained matcher.
func CompressWithMatcher(lines []string, matcher *drain3.Matcher) (DrainCompressed, error) {
	c := &compressed{
		templates:   matcher.Templates(),
		lineCount:   len(lines),
		bitmap:      bitset.New(uint(len(lines))),
		templateIDs: make([]int, 0, len(lines)),
		args:        make([][]string, 0, len(lines)),
	}

	// Build template lookup for ValueAt.
	c.tmplByID = make(map[int]drain3.Template, len(c.templates))
	for _, t := range c.templates {
		c.tmplByID[t.ID] = t
	}

	var buf [32]string
	for i, line := range lines {
		templateID, args, ok := matcher.MatchInto(line, buf[:0])
		if !ok {
			c.unmatched = append(c.unmatched, line)
			continue
		}
		c.bitmap.Set(uint(i))
		c.templateIDs = append(c.templateIDs, templateID)
		c.args = append(c.args, append([]string(nil), args...))
	}

	return c, nil
}

type compressed struct {
	templates   []drain3.Template
	tmplByID    map[int]drain3.Template
	lineCount   int
	bitmap      *bitset.BitSet
	templateIDs []int
	args        [][]string
	unmatched   []string
}

// ValueAt reconstructs the string at position i. Panics if i is out of range.
func (c *compressed) ValueAt(i int) string {
	if i < 0 || i >= c.lineCount {
		panic(fmt.Sprintf("compress: index %d out of range [0, %d)", i, c.lineCount))
	}
	ui := uint(i)
	if c.bitmap.Test(ui) {
		matchIdx := int(c.bitmap.Rank(ui)) - 1
		tmpl := c.tmplByID[c.templateIDs[matchIdx]]
		return reconstruct(tmpl, c.args[matchIdx])
	}
	unmatchIdx := i - int(c.bitmap.Rank(ui))
	return c.unmatched[unmatchIdx]
}

const (
	compressMagic   = "dc01"
	compressVersion = byte(1)
)

// WriteTo serializes the compressed representation to w.
func (c *compressed) WriteTo(w io.Writer) (int64, error) {
	var buf bytes.Buffer

	buf.WriteString(compressMagic)
	buf.WriteByte(compressVersion)

	writeUvarint(&buf, uint64(c.lineCount))

	// Bitmap.
	words := c.bitmap.Words()
	writeUvarint(&buf, uint64(len(words)))
	for _, word := range words {
		binary.Write(&buf, binary.LittleEndian, word)
	}

	// Templates.
	writeUvarint(&buf, uint64(len(c.templates)))
	for _, t := range c.templates {
		writeVarint(&buf, int64(t.ID))
		writeVarint(&buf, int64(t.Count))
		writeUvarint(&buf, uint64(t.TokenCount))

		paramWords := t.Params.Words()
		writeUvarint(&buf, uint64(len(paramWords)))
		for _, pw := range paramWords {
			binary.Write(&buf, binary.LittleEndian, pw)
		}

		writeUvarint(&buf, uint64(len(t.Tokens)))
		for _, tok := range t.Tokens {
			writeString(&buf, tok)
		}
	}

	// TemplateIDs.
	writeUvarint(&buf, uint64(len(c.templateIDs)))
	for _, id := range c.templateIDs {
		writeVarint(&buf, int64(id))
	}

	// Args.
	writeUvarint(&buf, uint64(len(c.args)))
	for _, args := range c.args {
		writeUvarint(&buf, uint64(len(args)))
		for _, a := range args {
			writeString(&buf, a)
		}
	}

	// Unmatched.
	writeUvarint(&buf, uint64(len(c.unmatched)))
	for _, s := range c.unmatched {
		writeString(&buf, s)
	}

	n, err := buf.WriteTo(w)
	return n, err
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
	ver, err := br.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("compress: read version: %w", err)
	}
	if ver != compressVersion {
		return nil, fmt.Errorf("compress: unsupported version %d", ver)
	}

	lineCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read line count: %w", err)
	}

	// Bitmap.
	wordCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read bitmap word count: %w", err)
	}
	words := make([]uint64, wordCount)
	for j := range words {
		if err := binary.Read(br, binary.LittleEndian, &words[j]); err != nil {
			return nil, fmt.Errorf("compress: read bitmap word: %w", err)
		}
	}
	bm := bitset.From(words)

	// Templates.
	tmplCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read template count: %w", err)
	}
	templates := make([]drain3.Template, tmplCount)
	for i := range templates {
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

		pwCount, err := readUvarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read param word count: %w", err)
		}
		pwords := make([]uint64, pwCount)
		for j := range pwords {
			if err := binary.Read(br, binary.LittleEndian, &pwords[j]); err != nil {
				return nil, fmt.Errorf("compress: read param word: %w", err)
			}
		}

		denseCount, err := readUvarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read dense token count: %w", err)
		}
		tokens := make([]string, denseCount)
		for j := range tokens {
			tokens[j], err = readString(br)
			if err != nil {
				return nil, fmt.Errorf("compress: read token: %w", err)
			}
		}

		templates[i] = drain3.Template{
			ID:         int(id),
			Tokens:     tokens,
			Params:     bitset.From(pwords),
			TokenCount: int(tokenCount),
			Count:      int(count),
		}
	}

	// TemplateIDs.
	matchCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read match count: %w", err)
	}
	templateIDs := make([]int, matchCount)
	for i := range templateIDs {
		v, err := readVarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read template id: %w", err)
		}
		templateIDs[i] = int(v)
	}

	// Args.
	argsCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read args count: %w", err)
	}
	args := make([][]string, argsCount)
	for i := range args {
		n, err := readUvarint(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read arg count: %w", err)
		}
		args[i] = make([]string, n)
		for j := range args[i] {
			args[i][j], err = readString(br)
			if err != nil {
				return nil, fmt.Errorf("compress: read arg: %w", err)
			}
		}
	}

	// Unmatched.
	unmatchedCount, err := readUvarint(br)
	if err != nil {
		return nil, fmt.Errorf("compress: read unmatched count: %w", err)
	}
	unmatched := make([]string, unmatchedCount)
	for i := range unmatched {
		unmatched[i], err = readString(br)
		if err != nil {
			return nil, fmt.Errorf("compress: read unmatched: %w", err)
		}
	}

	tmplByID := make(map[int]drain3.Template, len(templates))
	for _, t := range templates {
		tmplByID[t.ID] = t
	}

	return &compressed{
		templates:   templates,
		tmplByID:    tmplByID,
		lineCount:   int(lineCount),
		bitmap:      bm,
		templateIDs: templateIDs,
		args:        args,
		unmatched:   unmatched,
	}, nil
}

func reconstruct(tmpl drain3.Template, args []string) string {
	var b strings.Builder
	denseIdx := 0
	argIdx := 0
	for i := range tmpl.TokenCount {
		if i > 0 {
			b.WriteByte(' ')
		}
		if tmpl.Params.Test(uint(i)) && argIdx < len(args) {
			b.WriteString(args[argIdx])
			argIdx++
		} else {
			b.WriteString(tmpl.Tokens[denseIdx])
			denseIdx++
		}
	}
	return b.String()
}

// Binary encoding helpers.

func writeUvarint(w *bytes.Buffer, v uint64) {
	var scratch [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(scratch[:], v)
	w.Write(scratch[:n])
}

func writeVarint(w *bytes.Buffer, v int64) {
	var scratch [binary.MaxVarintLen64]byte
	n := binary.PutVarint(scratch[:], v)
	w.Write(scratch[:n])
}

func writeString(w *bytes.Buffer, s string) {
	writeUvarint(w, uint64(len(s)))
	w.WriteString(s)
}

func readUvarint(r *bytes.Reader) (uint64, error) {
	return binary.ReadUvarint(r)
}

func readVarint(r *bytes.Reader) (int64, error) {
	return binary.ReadVarint(r)
}

func readString(r *bytes.Reader) (string, error) {
	n, err := binary.ReadUvarint(r)
	if err != nil {
		return "", err
	}
	if n > uint64(r.Len()) {
		return "", io.ErrUnexpectedEOF
	}
	buf := make([]byte, int(n))
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}
