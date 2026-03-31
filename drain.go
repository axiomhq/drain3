package drain3

import (
	"slices"
	"strings"

	"github.com/bits-and-blooms/bitset"
)

// Template is a trained log template.
type Template struct {
	ID         int
	Tokens     []string       // dense: only non-param tokens, in order
	Params     *bitset.BitSet // bit i set = position i is a param placeholder
	TokenCount int            // total number of positions (len(Tokens) + Params.Count())
	Count      int            // number of matching log lines
}

// Matcher matches logs to trained Drain templates.
//
// Tokenization splits on individual space characters, not whitespace runs.
// Consecutive spaces produce empty tokens, so "a  b" (two spaces) and "a b"
// (one space) have different token counts and will never match each other.
// This preserves the exact whitespace layout for lossless reconstruction.
type Matcher struct {
	// Serialized state (persisted by MarshalBinary / restored by UnmarshalBinary).
	cfg       Config
	templates []Template

	// Rebuilt indexes — derived from cfg + templates by rebuildFromTemplates.
	rootByLen []*node    // prefix tree keyed by token count
	clusters  []*cluster // cluster ID → cluster, 0 is sentinel
	dictIDs     map[string]uint64 // token string → numeric ID
	dictNextID  uint64            // next token ID to assign
	paramID     uint64            // numeric ID of cfg.ParamString
	nextCluster int               // next cluster ID to assign

	// Scratch buffers — reused across addLogMessage calls during training.
	scratchIDs []uint64
	scratchTok []string
}

type node struct {
	children   map[uint64]*node
	clusterIDs []int
}

type cluster struct {
	id         int
	size       int
	paramCount int
	tokenIDs   []uint64
	tokenStr   []string
}

func newCluster(id int, tokenStr []string, tokenIDs []uint64, size int, paramID uint64) *cluster {
	c := &cluster{id: id, size: size, tokenStr: tokenStr, tokenIDs: tokenIDs}
	for _, tid := range tokenIDs {
		if tid == paramID {
			c.paramCount++
		}
	}
	return c
}

func (c *cluster) extractArgsInto(lineTokens []string, paramID uint64, dst []string) []string {
	if len(c.tokenIDs) == 0 || len(lineTokens) == 0 || c.paramCount == 0 {
		return nil
	}
	limit := min(len(c.tokenIDs), len(lineTokens))
	args := dst[:0]
	if cap(args) < min(c.paramCount, limit) {
		args = make([]string, 0, c.paramCount)
	}
	for i := 0; i < limit; i++ {
		if c.tokenIDs[i] == paramID {
			args = append(args, lineTokens[i])
		}
	}
	if len(args) == 0 {
		return nil
	}
	return args
}

func newMatcher(cfg Config) *Matcher {
	m := &Matcher{
		cfg:         cfg,
		clusters:    make([]*cluster, 1),
		nextCluster: 1,
		dictIDs:     make(map[string]uint64),
		dictNextID:  1,
	}
	m.paramID = m.internToken(cfg.ParamString)
	return m
}

func newNode() *node {
	return &node{children: make(map[uint64]*node)}
}

// Config returns matcher configuration.
func (m *Matcher) Config() Config {
	if m == nil {
		return Config{}
	}
	cfg := m.cfg
	if len(cfg.ExtraDelimiters) > 0 {
		cfg.ExtraDelimiters = append([]string(nil), cfg.ExtraDelimiters...)
	}
	return cfg
}

// Templates returns trained templates.
func (m *Matcher) Templates() []Template {
	if m == nil {
		return nil
	}
	return deepCopyTemplates(m.templates)
}

func (m *Matcher) internTokenIDs(tokens []string, dst []uint64) []uint64 {
	if len(tokens) == 0 {
		return nil
	}
	if cap(dst) < len(tokens) {
		dst = make([]uint64, len(tokens))
	} else {
		dst = dst[:len(tokens)]
	}
	for i, tok := range tokens {
		dst[i] = m.internToken(tok)
	}
	return dst
}

func (m *Matcher) internToken(token string) uint64 {
	if id, ok := m.dictIDs[token]; ok {
		return id
	}
	id := m.dictNextID
	m.dictNextID++
	m.dictIDs[token] = id
	return id
}


func tokenize(content string, extraDelimiters []string) []string {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}
	for _, delimiter := range extraDelimiters {
		content = strings.ReplaceAll(content, delimiter, " ")
	}
	return strings.Fields(content)
}

// tokenizeWhitespaceInto splits content on each individual space character.
// Unlike strings.Fields, consecutive spaces produce empty-string tokens,
// preserving the original whitespace layout for exact reconstruction.
func tokenizeWhitespaceInto(content string, dst []string) []string {
	if content == "" {
		return nil
	}
	if dst != nil {
		dst = dst[:0]
	}
	// strings.IndexByte uses SIMD but pays per-call overhead (function call +
	// string re-slice). For short strings (≤128 bytes) with frequent spaces
	// the overhead dominates; a scalar byte loop is faster. For longer
	// strings the SIMD scan between spaces amortises the cost.
	if len(content) <= 128 {
		start := 0
		for i := 0; i < len(content); i++ {
			if content[i] == ' ' {
				dst = append(dst, content[start:i])
				start = i + 1
			}
		}
		dst = append(dst, content[start:])
		return dst
	}
	for {
		i := strings.IndexByte(content, ' ')
		if i < 0 {
			break
		}
		dst = append(dst, content[:i])
		content = content[i+1:]
	}
	dst = append(dst, content)
	return dst
}

func hasNumbers(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			return true
		}
	}
	return false
}

func deepCopyTemplates(in []Template) []Template {
	if len(in) == 0 {
		return nil
	}
	out := make([]Template, len(in))
	for i := range in {
		out[i] = Template{
			ID:         in[i].ID,
			Tokens:     slices.Clone(in[i].Tokens),
			Params:     in[i].Params.Clone(),
			TokenCount: in[i].TokenCount,
			Count:      in[i].Count,
		}
	}
	return out
}
