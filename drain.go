package drain3

import (
	"encoding/binary"
	"math/bits"
	"slices"
	"strings"
	"unsafe"

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
	prefilterBuckets []prefilterBucket          // match prefilter indexes, keyed by token count
	matchNeeded      []int                      // precomputed ceil(MatchThreshold * tokenCount), keyed by token count

	hasParamFirst bool // true if any cluster has paramID at position 0

	maxProbe int // widest probePos across prefilter buckets; sizes Session.probeIDs

	// Training-only scratch. Match-path scratch lives on Session.
	scratchIDs []uint64
	scratchTok []string

	// defaultSession backs the Matcher-level Match* methods, preserving
	// the historical one-goroutine-per-Matcher contract for them.
	defaultSession *Session
}

type prefilterBucket struct {
	// Normal Match prefilter groups candidates by first and/or last token.
	any       []int    // templates whose first and last tokens are params
	firstKeys []uint64 // sorted literal first-token IDs
	firstVals [][]int  // candidate template IDs for firstKeys
	lastKeys  []uint64 // sorted literal last-token IDs
	lastVals  [][]int  // candidate template IDs for lastKeys
	flKeys    []uint64 // sorted packed first/last-token ID pairs
	flVals    [][]int  // candidate template IDs for flKeys
	exactAny  []int    // exact-match templates with no literal anchor tokens

	// Exact Match prefilter groups candidates by first/last non-param
	// anchors. Each distinct anchor position resolves to a dictionary ID
	// once per line into a probePos slot; single- and pair-anchor groups
	// then binary-search their per-position ID range. Pair IDs are packed
	// (id0<<32)|id1 like flKeys — a collision only adds false candidates,
	// which the scorer rejects.
	probePos        []uint16    // distinct anchor positions, resolved once per line
	anchorSlot      []uint16    // per single-anchor position: slot into probePos
	anchorStart     []int32     // len(anchorSlot)+1 prefix ranges into anchorIDs
	anchorIDs       []uint64    // sorted anchor-token IDs within each position range
	anchorVals      [][]int     // candidate template IDs for anchorIDs
	anchorPairSlot  [][2]uint16 // per anchor-position pair: slots into probePos
	anchorPairStart []int32     // len(anchorPairSlot)+1 prefix ranges into anchorPairIDs
	anchorPairIDs   []uint64    // sorted packed anchor-ID pairs within each range
	anchorPairVals  [][]int     // candidate template IDs for anchorPairIDs
}

// anchorKey and anchorPairKey are build-time accumulator keys for
// rebuildMatchPrefilter; the probe path uses the flattened slot/range
// representation on prefilterBucket.
type anchorKey struct {
	pos uint16
	id  uint64
}

type anchorPairKey struct {
	pos0 uint16
	pos1 uint16
	id0  uint64
	id1  uint64
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
	c.paramCount = 0
	for i, tid := range c.tokenIDs {
		if tid == paramID {
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

// appendArgs appends the tokens at c's param positions to dst and
// returns it. The appended strings alias lineTokens' backing line.
func (c *cluster) appendArgs(dst []string, lineTokens []string, paramID uint64) []string {
	limit := min(len(c.tokenIDs), len(lineTokens))
	for i := 0; i < limit; i++ {
		if c.tokenIDs[i] == paramID {
			dst = append(dst, lineTokens[i])
		}
	}
	return dst
}

func (c *cluster) extractArgsInto(lineTokens []string, paramID uint64, dst []string) []string {
	if len(c.tokenIDs) == 0 || len(lineTokens) == 0 || c.paramCount == 0 {
		return nil
	}
	args := dst[:0]
	if cap(args) < min(c.paramCount, min(len(c.tokenIDs), len(lineTokens))) {
		args = make([]string, 0, c.paramCount)
	}
	args = c.appendArgs(args, lineTokens, paramID)
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
	m.defaultSession = m.NewSession()
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
//
// Spaces are located with a SWAR scan, 8 bytes per load. The mask is the
// carry-free per-byte equality test — (x&^hi)+0x7f sets bit 7 of a byte
// iff its low 7 bits are nonzero and never carries across bytes — so
// every set bit marks a space exactly and one load serves all spaces in
// the window. Measured on M3 Max against real log lines (~12 B/token):
// 1.75x over the byte loop; a strings.IndexByte loop was a wash (call
// overhead per short token cancels the SIMD win).
func tokenizeWhitespaceCount(content string, dst []string, maxTokens int) ([]string, int) {
	if content == "" || maxTokens <= 0 {
		return dst[:0], 0
	}
	const (
		swarHi     = 0x8080808080808080
		swarLo7    = 0x7f7f7f7f7f7f7f7f
		swarSpaces = 0x2020202020202020
	)
	buf := unsafe.Slice(unsafe.StringData(content), len(content))
	n := len(content)
	dst = dst[:0]
	start := 0
	count := 1
	i := 0
	for ; i+8 <= n; i += 8 {
		x := binary.LittleEndian.Uint64(buf[i:]) ^ swarSpaces
		m := ^(((x &^ swarHi) + swarLo7) | x) & swarHi
		for m != 0 {
			j := i + bits.TrailingZeros64(m)>>3
			m &= m - 1
			dst = append(dst, content[start:j])
			start = j + 1
			count++
			if count > maxTokens {
				return dst, count
			}
		}
	}
	for ; i < n; i++ {
		if content[i] == ' ' {
			dst = append(dst, content[start:i])
			start = i + 1
			count++
			if count > maxTokens {
				return dst, count
			}
		}
	}
	return append(dst, content[start:]), count
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
