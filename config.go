package drain3

import "fmt"

// MatchScorer selects how the prefilter path verifies candidate templates.
type MatchScorer int

const (
	// MatchScorerAuto (zero value) picks String or ID at build time from
	// the trained template count: small dictionaries use String (no
	// regression), large ones use ID (the up-front token-ID resolution
	// amortizes across the larger candidate scans). See autoScorerTemplateThreshold.
	MatchScorerAuto MatchScorer = iota
	// MatchScorerString compares candidate tokens as strings. No speedup,
	// no regression on any workload. Best for small dictionaries or
	// miss-dominated streams.
	MatchScorerString
	// MatchScorerID resolves line tokens to dictionary IDs once and
	// compares uint64s. Large win for large-dictionary, hit-dominated
	// streams; a modest penalty on misses.
	MatchScorerID
)

// Config controls training and matching behavior.
type Config struct {
	Depth                    int
	SimilarityThreshold      float64
	MatchThreshold           float64
	MaxChildren              int
	MaxTokens                int
	MaxBytes                 int
	MaxClusters              int // 0 means unlimited
	ParamString              string
	ParametrizeNumericTokens bool
	ExtraDelimiters          []string
	EnableMatchPrefilter     bool
	MatchScorer              MatchScorer // zero value = MatchScorerAuto
}

// autoScorerTemplateThreshold is the trained-template count at or above
// which MatchScorerAuto resolves to MatchScorerID. Below it, ID's up-front
// per-line token resolution does not amortize over the small candidate
// scans, so String is used. Heuristic: small dictionaries (hundreds) favor
// String; larger ones (≈thousands) favor ID by a wide margin.
const autoScorerTemplateThreshold = 1024

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
	if cfg.Depth < 3 {
		return Config{}, fmt.Errorf("depth must be >= 3, got %d", cfg.Depth)
	}
	if cfg.SimilarityThreshold < 0 || cfg.SimilarityThreshold > 1 {
		return Config{}, fmt.Errorf("similarity threshold must be in [0, 1], got %f", cfg.SimilarityThreshold)
	}
	if cfg.MatchThreshold < 0 || cfg.MatchThreshold > 1 {
		return Config{}, fmt.Errorf("match threshold must be in [0, 1], got %f", cfg.MatchThreshold)
	}
	if cfg.MaxChildren < 2 {
		return Config{}, fmt.Errorf("max children must be >= 2, got %d", cfg.MaxChildren)
	}
	if cfg.MaxTokens < 1 {
		return Config{}, fmt.Errorf("max tokens must be >= 1, got %d", cfg.MaxTokens)
	}
	if cfg.MaxBytes < 1 {
		return Config{}, fmt.Errorf("max bytes must be >= 1, got %d", cfg.MaxBytes)
	}
	if cfg.MaxClusters < 0 {
		return Config{}, fmt.Errorf("max clusters must be >= 0, got %d", cfg.MaxClusters)
	}
	if cfg.ParamString == "" {
		return Config{}, fmt.Errorf("param string must not be empty")
	}
	if cfg.MatchScorer < MatchScorerAuto || cfg.MatchScorer > MatchScorerID {
		return Config{}, fmt.Errorf("invalid match scorer %d", cfg.MatchScorer)
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
