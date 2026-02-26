package drain3

import "fmt"

// Config controls training and matching behavior.
type Config struct {
	Depth                    int
	SimilarityThreshold      float64
	MatchThreshold           float64
	MaxChildren              int
	MaxTokens                int
	MaxBytes                 int
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
		cfg.ParamString == "" &&
		!cfg.ParametrizeNumericTokens &&
		!cfg.EnableMatchPrefilter &&
		len(cfg.ExtraDelimiters) == 0
}
