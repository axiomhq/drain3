package drain3

import (
	"slices"
	"strings"

	"github.com/bits-and-blooms/bitset"
	"github.com/lemire/constmap"
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
	// Core state. To round-trip a trained matcher, export templates via
	// Templates() and rebuild with NewMatcherFromTemplates.
	cfg       Config
	templates []Template

	// Derived indexes — rebuilt from cfg + templates by rebuildFromTemplates.
	rootByLen        []*node                    // prefix tree keyed by token count
	clusters         []*cluster                 // cluster ID → cluster, 0 is sentinel
	dictIDs          map[string]uint64          // token string → numeric ID (used during training)
	dictFrozen       *constmap.VerifiedConstMap // read-only lookup table built after training
	dictNextID       uint64                     // next token ID to assign
	paramID          uint64                     // numeric ID of cfg.ParamString
	nextCluster      int                        // next cluster ID to assign
	prefilterBuckets []prefilterBucket          // first/last-token prefilter index, keyed by token count
	matchNeeded      []int                      // precomputed ceil(MatchThreshold * tokenCount), keyed by token count

	hasParamFirst bool // true if any cluster has paramID at position 0

	// Scratch buffers — reused across calls.
	scratchIDs        []uint64
	scratchTok        []string
	scratchCandidates []int
}

type prefilterBucket struct {
	any       []int
	firstKeys []uint64
	firstVals [][]int
	lastKeys  []uint64
	lastVals  [][]int
	flKeys    []uint64
	flVals    [][]int
}

type node struct {
	children   map[uint64]*node
	clusterIDs []int
}

type cluster struct {
	id          int
	size        int
	paramCount  int
	tokenIDs    []uint64
	tokenStr    []string
	nonParamIdx []uint16
	paramIdx    []uint16 // positions where tokenIDs[i] == paramID; complement of nonParamIdx
	// Anchor positions for cheap pre-rejection. -1 = no anchor.
	// anchor0 is the first non-param position, anchor1 is the last.
	anchor0 int
	anchor1 int
}

func newCluster(id int, tokenStr []string, tokenIDs []uint64, size int, paramID uint64) *cluster {
	c := &cluster{id: id, size: size, tokenStr: tokenStr, tokenIDs: tokenIDs}
	c.buildNonParamIdx(paramID)
	return c
}

func (c *cluster) buildNonParamIdx(paramID uint64) {
	c.nonParamIdx = c.nonParamIdx[:0]
	c.paramIdx = c.paramIdx[:0]
	c.paramCount = 0
	for i, tid := range c.tokenIDs {
		if tid == paramID {
			c.paramIdx = append(c.paramIdx, uint16(i))
			c.paramCount++
		} else {
			c.nonParamIdx = append(c.nonParamIdx, uint16(i))
		}
	}
	if len(c.nonParamIdx) >= 2 {
		c.anchor0 = int(c.nonParamIdx[0])
		c.anchor1 = int(c.nonParamIdx[len(c.nonParamIdx)-1])
	} else if len(c.nonParamIdx) == 1 {
		c.anchor0 = int(c.nonParamIdx[0])
		c.anchor1 = -1
	} else {
		c.anchor0 = -1
		c.anchor1 = -1
	}
}

func (c *cluster) rebuildNonParamIdx(paramID uint64) {
	c.buildNonParamIdx(paramID)
}

func (c *cluster) extractArgsInto(lineTokens []string, dst []string) []string {
	if c.paramCount == 0 || len(lineTokens) == 0 {
		return nil
	}
	nLine := len(lineTokens)
	args := dst[:0]
	if cap(args) < c.paramCount {
		args = make([]string, 0, c.paramCount)
	}
	for _, i := range c.paramIdx {
		pos := int(i)
		if pos >= nLine {
			break
		}
		args = append(args, lineTokens[pos])
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

func (m *Matcher) freezeDict() {
	keys := make([]string, 0, len(m.dictIDs))
	vals := make([]uint64, 0, len(m.dictIDs))
	for k, v := range m.dictIDs {
		keys = append(keys, k)
		vals = append(vals, v)
	}
	vm, err := constmap.NewVerified(keys, vals)
	if err != nil {
		panic("drain3: failed to build constmap: " + err.Error())
	}
	m.dictFrozen = vm
	m.scratchCandidates = make([]int, 0, 1024)
	if cap(m.scratchTok) < m.cfg.MaxTokens {
		m.scratchTok = make([]string, 0, m.cfg.MaxTokens)
	}
	// Check if any cluster has a param at position 0. If so, we can't
	// reject lines based on an unknown first token alone.
	m.hasParamFirst = false
	for _, c := range m.clusters {
		if c != nil && len(c.tokenIDs) > 0 && c.tokenIDs[0] == m.paramID {
			m.hasParamFirst = true
			break
		}
	}
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

// tokenizeWhitespaceCount splits on spaces and returns the token count in
// a single pass, eliminating the separate strings.Count call.
// maxTokens limits scanning: if the count would exceed maxTokens the
// function returns early with a count > maxTokens so the caller can reject.
func tokenizeWhitespaceCount(content string, dst []string, maxTokens int) ([]string, int) {
	if content == "" || maxTokens <= 0 {
		return dst[:0], 0
	}

	dst = dst[:0]
	start := 0
	count := 1

	for i := 0; i < len(content); i++ {
		if content[i] != ' ' {
			continue
		}
		dst = append(dst, content[start:i])
		start = i + 1
		count++
		if count > maxTokens {
			return dst, count
		}
	}

	dst = append(dst, content[start:])
	return dst, count
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
