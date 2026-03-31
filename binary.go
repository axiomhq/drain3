package drain3

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/bits-and-blooms/bitset"
)

const (
	payloadMagic   = "drn3"
	payloadVersion = byte(4)
)

// MarshalBinary serializes the matcher into a binary format.
func (m *Matcher) MarshalBinary() ([]byte, error) {
	if m == nil {
		return nil, errors.New("nil matcher")
	}

	var buf bytes.Buffer
	buf.WriteString(payloadMagic)
	buf.WriteByte(payloadVersion)

	writeConfigBinary(&buf, m.cfg)

	templates := m.templates
	writeUvarint(&buf, uint64(len(templates)))
	for _, t := range templates {
		writeVarint(&buf, int64(t.ID))
		writeVarint(&buf, int64(t.Count))
		writeUvarint(&buf, uint64(t.TokenCount))
		words := t.Params.Words()
		writeUvarint(&buf, uint64(len(words)))
		for _, w := range words {
			writeUint64(&buf, w)
		}
		writeUvarint(&buf, uint64(len(t.Tokens)))
		for _, tok := range t.Tokens {
			writeString(&buf, tok)
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary restores a matcher from binary data produced by MarshalBinary.
func (m *Matcher) UnmarshalBinary(data []byte) error {
	if m == nil {
		return errors.New("nil matcher")
	}

	r := bytes.NewReader(data)
	magic := make([]byte, len(payloadMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != payloadMagic {
		return errors.New("invalid payload magic")
	}
	ver, err := r.ReadByte()
	if err != nil {
		return fmt.Errorf("read version: %w", err)
	}
	if ver != payloadVersion {
		return fmt.Errorf("unsupported payload version: %d", ver)
	}

	cfg, err := readConfigBinary(r)
	if err != nil {
		return err
	}

	n, err := binary.ReadUvarint(r)
	if err != nil {
		return fmt.Errorf("read template count: %w", err)
	}
	if n > uint64(^uint(0)>>1) {
		return errors.New("template count overflows int")
	}

	templates := make([]Template, int(n))
	for i := range templates {
		id, err := binary.ReadVarint(r)
		if err != nil {
			return fmt.Errorf("read template id: %w", err)
		}
		count, err := binary.ReadVarint(r)
		if err != nil {
			return fmt.Errorf("read template count: %w", err)
		}
		tokenCount, err := binary.ReadUvarint(r)
		if err != nil {
			return fmt.Errorf("read token count: %w", err)
		}
		if tokenCount > uint64(^uint(0)>>1) {
			return errors.New("token count overflows int")
		}

		var tmpl Template
		tmpl.ID = int(id)
		tmpl.Count = int(count)
		tmpl.TokenCount = int(tokenCount)

		wordCount, err := binary.ReadUvarint(r)
		if err != nil {
			return fmt.Errorf("read bitset word count: %w", err)
		}
		if wordCount > uint64(^uint(0)>>1) {
			return errors.New("bitset word count overflows int")
		}
		words := make([]uint64, int(wordCount))
		for j := range words {
			words[j], err = readUint64(r)
			if err != nil {
				return fmt.Errorf("read bitset word: %w", err)
			}
		}
		tmpl.Params = bitset.From(words)

		denseCount, err := binary.ReadUvarint(r)
		if err != nil {
			return fmt.Errorf("read dense token count: %w", err)
		}
		if denseCount > uint64(^uint(0)>>1) {
			return errors.New("dense token count overflows int")
		}
		tmpl.Tokens = make([]string, int(denseCount))
		for j := range tmpl.Tokens {
			tok, err := readString(r)
			if err != nil {
				return fmt.Errorf("read token: %w", err)
			}
			tmpl.Tokens[j] = tok
		}

		templates[i] = tmpl
	}

	return m.rebuildFromTemplates(cfg, templates)
}

// LoadMatcher deserializes a matcher from binary data produced by MarshalBinary.
func LoadMatcher(data []byte) (*Matcher, error) {
	var m Matcher
	if err := m.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	return &m, nil
}

func writeConfigBinary(w *bytes.Buffer, cfg Config) {
	writeInt32(w, int32(cfg.Depth))
	writeFloat64(w, cfg.SimilarityThreshold)
	writeFloat64(w, cfg.MatchThreshold)
	writeInt32(w, int32(cfg.MaxChildren))
	writeInt32(w, int32(cfg.MaxTokens))
	writeInt32(w, int32(cfg.MaxBytes))
	writeVarint(w, int64(cfg.MaxClusters))
	writeString(w, cfg.ParamString)
	if cfg.ParametrizeNumericTokens {
		w.WriteByte(1)
	} else {
		w.WriteByte(0)
	}
	writeUvarint(w, uint64(len(cfg.ExtraDelimiters)))
	for _, d := range cfg.ExtraDelimiters {
		writeString(w, d)
	}
	if cfg.EnableMatchPrefilter {
		w.WriteByte(1)
	} else {
		w.WriteByte(0)
	}
}

func readConfigBinary(r *bytes.Reader) (Config, error) {
	depth, err := readInt32(r)
	if err != nil {
		return Config{}, fmt.Errorf("read depth: %w", err)
	}
	simTh, err := readFloat64(r)
	if err != nil {
		return Config{}, fmt.Errorf("read similarity threshold: %w", err)
	}
	matchTh, err := readFloat64(r)
	if err != nil {
		return Config{}, fmt.Errorf("read match threshold: %w", err)
	}
	maxChildren, err := readInt32(r)
	if err != nil {
		return Config{}, fmt.Errorf("read max children: %w", err)
	}
	maxTokens, err := readInt32(r)
	if err != nil {
		return Config{}, fmt.Errorf("read max tokens: %w", err)
	}
	maxBytes, err := readInt32(r)
	if err != nil {
		return Config{}, fmt.Errorf("read max bytes: %w", err)
	}
	maxClusters, err := binary.ReadVarint(r)
	if err != nil {
		return Config{}, fmt.Errorf("read max clusters: %w", err)
	}
	param, err := readString(r)
	if err != nil {
		return Config{}, err
	}
	flag, err := r.ReadByte()
	if err != nil {
		return Config{}, fmt.Errorf("read numeric parameterization flag: %w", err)
	}
	nDelims, err := binary.ReadUvarint(r)
	if err != nil {
		return Config{}, fmt.Errorf("read delimiter count: %w", err)
	}
	if nDelims > uint64(^uint(0)>>1) {
		return Config{}, errors.New("delimiter count overflows int")
	}
	var delims []string
	if nDelims > 0 {
		delims = make([]string, int(nDelims))
		for i := range delims {
			d, err := readString(r)
			if err != nil {
				return Config{}, err
			}
			delims[i] = d
		}
	}

	prefilterFlag, err := r.ReadByte()
	if err != nil {
		return Config{}, fmt.Errorf("read match prefilter flag: %w", err)
	}

	cfg := Config{
		Depth:                    int(depth),
		SimilarityThreshold:      simTh,
		MatchThreshold:           matchTh,
		MaxChildren:              int(maxChildren),
		MaxTokens:                int(maxTokens),
		MaxBytes:                 int(maxBytes),
		MaxClusters:              int(maxClusters),
		ParamString:              param,
		ParametrizeNumericTokens: flag == 1,
		ExtraDelimiters:          delims,
		EnableMatchPrefilter:     prefilterFlag == 1,
	}
	return normalizeConfig(cfg)
}

func writeString(w *bytes.Buffer, s string) {
	writeUvarint(w, uint64(len(s)))
	w.WriteString(s)
}

func readString(r *bytes.Reader) (string, error) {
	n, err := binary.ReadUvarint(r)
	if err != nil {
		return "", fmt.Errorf("read string length: %w", err)
	}
	if n > uint64(r.Len()) {
		return "", io.ErrUnexpectedEOF
	}
	buf := make([]byte, int(n))
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("read string bytes: %w", err)
	}
	return string(buf), nil
}

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

func writeInt32(w *bytes.Buffer, v int32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(v))
	w.Write(b[:])
}

func writeFloat64(w *bytes.Buffer, v float64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
	w.Write(b[:])
}

func writeUint64(w *bytes.Buffer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.Write(b[:])
}

func readInt32(r *bytes.Reader) (int32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return int32(binary.LittleEndian.Uint32(b[:])), nil
}

func readFloat64(r *bytes.Reader) (float64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(b[:])), nil
}

func readUint64(r *bytes.Reader) (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}
