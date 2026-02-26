package drain3

import (
	"slices"
	"strings"
)

// Template is a trained log template.
type Template struct {
	ID     int
	Tokens []string
	Count  int
}

// Matcher matches logs to trained Drain templates.
type Matcher struct {
	// Serialized state (persisted by MarshalBinary / restored by UnmarshalBinary).
	cfg       Config
	templates []Template

	// Rebuilt indexes — derived from cfg + templates by rebuildFromTemplates.
	rootByLen        []*Node           // mutable prefix tree used during training
	flatRootByLen    []uint32          // read-only flat tree: root node index per token count
	flatNodes        []FlatNode        // read-only flat tree: all flattened nodes
	prefilterBuckets []prefilterBucket // per-token-count candidate index
	clusters         []*cluser         // cluster ID → cluster, 0 is sentinel
	dictIDs          map[string]uint64 // token string → numeric ID
	dictNextID       uint64            // next token ID to assign
	paramID          uint64            // numeric ID of cfg.ParamString
	nextCluster      int               // next cluster ID to assign
	matchNeeded      []int             // pre-computed required-score per token count

	// Scratch buffers — reused across training calls to reduce allocations.
	scratchIDs []uint64
	scratchTok []string
}

type Node struct {
	children   map[uint64]*Node
	clusterIDs []int
}

type FlatNode struct {
	childKeys    []uint64
	childIndexes []uint32
	clusterIDs   []int
}

type prefilterBucket struct {
	any []int // clusters with both first and last as param

	// Sorted arrays for binary-searched edge lookups.
	firstKeys []uint64
	firstVals [][]int
	lastKeys  []uint64
	lastVals  [][]int

	// Sorted array of (firstID<<32 | lastID) for combined lookup.
	flKeys []uint64
	flVals [][]int
}

func newMatcher(cfg Config) *Matcher {
	m := &Matcher{
		cfg:         cfg,
		clusters:    make([]*cluser, 1),
		nextCluster: 1,
		dictIDs:     make(map[string]uint64),
		dictNextID:  1,
	}
	m.paramID = m.internToken(cfg.ParamString)
	m.rebuildMatchNeeded()
	return m
}

func newNode() *Node {
	return &Node{children: make(map[uint64]*Node)}
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
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
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
			ID:     in[i].ID,
			Tokens: slices.Clone(in[i].Tokens),
			Count:  in[i].Count,
		}
	}
	return out
}
