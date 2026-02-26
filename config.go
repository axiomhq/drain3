package drain3

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// Config controls training and matching behavior.
type Config struct {
	Depth                    int
	SimilarityThreshold      float64
	MatchThreshold           float64
	MaxChildren              int
	MaxTokens                int
	MaxBytes                 int
	TopK                     int
	ParamString              string
	ParametrizeNumericTokens bool
	EnableMatchPrefilter     bool
	ExtraDelimiters          []string
}

// DefaultConfig returns default Drain settings.
func DefaultConfig() Config {
	return Config{
		Depth:                    4,
		SimilarityThreshold:      0.5,
		MatchThreshold:           1.0,
		MaxChildren:              100,
		MaxTokens:                64,
		MaxBytes:                 1024,
		TopK:                     250,
		ParamString:              "<*>",
		ParametrizeNumericTokens: true,
		EnableMatchPrefilter:     true,
	}
}

func normalizeConfig(cfg Config) (Config, error) {
	if isZeroConfig(cfg) {
		cfg = DefaultConfig()
	}

	def := DefaultConfig()
	if cfg.Depth == 0 {
		cfg.Depth = def.Depth
	}
	if cfg.SimilarityThreshold == 0 {
		cfg.SimilarityThreshold = def.SimilarityThreshold
	}
	if cfg.MatchThreshold == 0 {
		cfg.MatchThreshold = def.MatchThreshold
	}
	if cfg.MaxChildren == 0 {
		cfg.MaxChildren = def.MaxChildren
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = def.MaxTokens
	}
	if cfg.MaxBytes == 0 {
		cfg.MaxBytes = def.MaxBytes
	}
	if cfg.TopK == 0 {
		cfg.TopK = def.TopK
	}
	if cfg.ParamString == "" {
		cfg.ParamString = def.ParamString
	}

	if cfg.Depth < 3 {
		return Config{}, fmt.Errorf("depth must be >= 3")
	}
	if cfg.SimilarityThreshold < 0 || cfg.SimilarityThreshold > 1 {
		return Config{}, fmt.Errorf("similarity threshold must be in [0, 1]")
	}
	if cfg.MatchThreshold < 0 || cfg.MatchThreshold > 1 {
		return Config{}, fmt.Errorf("match threshold must be in [0, 1]")
	}
	if cfg.MaxChildren < 2 {
		return Config{}, fmt.Errorf("max children must be >= 2")
	}
	if cfg.MaxTokens < 1 {
		return Config{}, fmt.Errorf("max tokens must be >= 1")
	}
	if cfg.MaxBytes < 1 {
		return Config{}, fmt.Errorf("max bytes must be >= 1")
	}
	if cfg.TopK < 1 {
		return Config{}, fmt.Errorf("top k must be >= 1")
	}
	if cfg.ParamString == "" {
		return Config{}, fmt.Errorf("param string must not be empty")
	}

	if len(cfg.ExtraDelimiters) > 0 {
		filtered := make([]string, 0, len(cfg.ExtraDelimiters))
		for _, d := range cfg.ExtraDelimiters {
			if d != "" {
				filtered = append(filtered, d)
			}
		}
		cfg.ExtraDelimiters = filtered
	}

	return cfg, nil
}

func isZeroConfig(cfg Config) bool {
	return cfg.Depth == 0 &&
		cfg.SimilarityThreshold == 0 &&
		cfg.MatchThreshold == 0 &&
		cfg.MaxChildren == 0 &&
		cfg.MaxTokens == 0 &&
		cfg.MaxBytes == 0 &&
		cfg.TopK == 0 &&
		cfg.ParamString == "" &&
		!cfg.ParametrizeNumericTokens &&
		!cfg.EnableMatchPrefilter &&
		len(cfg.ExtraDelimiters) == 0
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
	var maxTokens, maxBytes, topK int32
	if ver >= 2 {
		if err := binary.Read(r, binary.LittleEndian, &maxTokens); err != nil {
			return Config{}, fmt.Errorf("read max tokens: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &maxBytes); err != nil {
			return Config{}, fmt.Errorf("read max bytes: %w", err)
		}
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
	nDelims, err := readUvarint(r)
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
		TopK:                     int(topK),
		ParamString:              param,
		ParametrizeNumericTokens: flag == 1,
		EnableMatchPrefilter:     prefilterFlag == 1,
		ExtraDelimiters:          delims,
	}
	return normalizeConfig(cfg)
}
