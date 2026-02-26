package drain3

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	payloadMagic   = "drn3"
	payloadVersion = byte(2)
)

func marshalBinary(m *Matcher) ([]byte, error) {
	if m == nil {
		return nil, errors.New("nil matcher")
	}

	var buf bytes.Buffer
	buf.WriteString(payloadMagic)
	buf.WriteByte(payloadVersion)

	if err := writeConfigBinary(&buf, m.cfg); err != nil {
		return nil, err
	}

	templates := m.templates
	if err := writeUvarint(&buf, uint64(len(templates))); err != nil {
		return nil, err
	}
	for _, t := range templates {
		if err := writeVarint(&buf, int64(t.ID)); err != nil {
			return nil, err
		}
		if err := writeVarint(&buf, int64(t.Count)); err != nil {
			return nil, err
		}
		if err := writeUvarint(&buf, uint64(len(t.Tokens))); err != nil {
			return nil, err
		}
		for _, tok := range t.Tokens {
			if err := writeString(&buf, tok); err != nil {
				return nil, err
			}
		}
	}

	return buf.Bytes(), nil
}

func unmarshalBinary(m *Matcher, data []byte) error {
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
	if ver != 1 && ver != 2 {
		return fmt.Errorf("unsupported payload version: %d", ver)
	}

	cfg, err := readConfigBinary(r, ver)
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
	for i := 0; i < len(templates); i++ {
		id, err := binary.ReadVarint(r)
		if err != nil {
			return fmt.Errorf("read template id: %w", err)
		}
		count, err := binary.ReadVarint(r)
		if err != nil {
			return fmt.Errorf("read template count: %w", err)
		}
		tokCount, err := binary.ReadUvarint(r)
		if err != nil {
			return fmt.Errorf("read token count: %w", err)
		}
		if tokCount > uint64(^uint(0)>>1) {
			return errors.New("token count overflows int")
		}

		tokens := make([]string, int(tokCount))
		for j := 0; j < len(tokens); j++ {
			tok, err := readString(r)
			if err != nil {
				return fmt.Errorf("read token: %w", err)
			}
			tokens[j] = tok
		}

		templates[i] = Template{ID: int(id), Tokens: tokens, Count: int(count)}
	}

	if err := m.rebuildFromTemplates(cfg, templates); err != nil {
		return err
	}

	return nil
}

func loadMatcher(data []byte) (*Matcher, error) {
	var m Matcher
	if err := unmarshalBinary(&m, data); err != nil {
		return nil, err
	}
	return &m, nil
}

func writeConfigBinary(w *bytes.Buffer, cfg Config) error {
	if err := binary.Write(w, binary.LittleEndian, int32(cfg.Depth)); err != nil {
		return fmt.Errorf("write depth: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, cfg.SimilarityThreshold); err != nil {
		return fmt.Errorf("write similarity threshold: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, cfg.MatchThreshold); err != nil {
		return fmt.Errorf("write match threshold: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int32(cfg.MaxChildren)); err != nil {
		return fmt.Errorf("write max children: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int32(cfg.MaxTokens)); err != nil {
		return fmt.Errorf("write max tokens: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int32(cfg.MaxBytes)); err != nil {
		return fmt.Errorf("write max bytes: %w", err)
	}
	// TopK was removed from Config but we write a placeholder to preserve the v2 wire format.
	if err := binary.Write(w, binary.LittleEndian, int32(0)); err != nil {
		return fmt.Errorf("write top k: %w", err)
	}
	if err := writeString(w, cfg.ParamString); err != nil {
		return err
	}
	if cfg.ParametrizeNumericTokens {
		w.WriteByte(1)
	} else {
		w.WriteByte(0)
	}
	if cfg.EnableMatchPrefilter {
		w.WriteByte(1)
	} else {
		w.WriteByte(0)
	}
	if err := writeUvarint(w, uint64(len(cfg.ExtraDelimiters))); err != nil {
		return err
	}
	for _, d := range cfg.ExtraDelimiters {
		if err := writeString(w, d); err != nil {
			return err
		}
	}
	return nil
}

func readConfigBinary(r *bytes.Reader, ver byte) (Config, error) {
	var depth int32
	if err := binary.Read(r, binary.LittleEndian, &depth); err != nil {
		return Config{}, fmt.Errorf("read depth: %w", err)
	}
	var simTh float64
	if err := binary.Read(r, binary.LittleEndian, &simTh); err != nil {
		return Config{}, fmt.Errorf("read similarity threshold: %w", err)
	}
	var matchTh float64
	if err := binary.Read(r, binary.LittleEndian, &matchTh); err != nil {
		return Config{}, fmt.Errorf("read match threshold: %w", err)
	}
	var maxChildren int32
	if err := binary.Read(r, binary.LittleEndian, &maxChildren); err != nil {
		return Config{}, fmt.Errorf("read max children: %w", err)
	}
	var maxTokens, maxBytes int32
	if ver >= 2 {
		if err := binary.Read(r, binary.LittleEndian, &maxTokens); err != nil {
			return Config{}, fmt.Errorf("read max tokens: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &maxBytes); err != nil {
			return Config{}, fmt.Errorf("read max bytes: %w", err)
		}
		// TopK was removed from Config but remains in the v2 wire format.
		var topK int32
		if err := binary.Read(r, binary.LittleEndian, &topK); err != nil {
			return Config{}, fmt.Errorf("read top k: %w", err)
		}
	}
	param, err := readString(r)
	if err != nil {
		return Config{}, err
	}
	flag, err := r.ReadByte()
	if err != nil {
		return Config{}, fmt.Errorf("read numeric parameterization flag: %w", err)
	}
	prefilterFlag, err := r.ReadByte()
	if err != nil {
		return Config{}, fmt.Errorf("read prefilter enable flag: %w", err)
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

	cfg := Config{
		Depth:                    int(depth),
		SimilarityThreshold:      simTh,
		MatchThreshold:           matchTh,
		MaxChildren:              int(maxChildren),
		MaxTokens:                int(maxTokens),
		MaxBytes:                 int(maxBytes),
		ParamString:              param,
		ParametrizeNumericTokens: flag == 1,
		EnableMatchPrefilter:     prefilterFlag == 1,
		ExtraDelimiters:          delims,
	}
	return normalizeConfig(cfg)
}

func writeString(w *bytes.Buffer, s string) error {
	if err := writeUvarint(w, uint64(len(s))); err != nil {
		return err
	}
	_, err := w.WriteString(s)
	if err != nil {
		return fmt.Errorf("write string bytes: %w", err)
	}
	return nil
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

func writeUvarint(w *bytes.Buffer, v uint64) error {
	var scratch [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(scratch[:], v)
	_, err := w.Write(scratch[:n])
	return err
}

func writeVarint(w *bytes.Buffer, v int64) error {
	var scratch [binary.MaxVarintLen64]byte
	n := binary.PutVarint(scratch[:], v)
	_, err := w.Write(scratch[:n])
	return err
}
