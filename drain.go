package drain3

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

const (
	payloadMagic                = "drn3"
	payloadVersion              = byte(1)
	adaptiveIDPathMinCandidates = 4
	maxMatchTokens = 128 // lines with more tokens skip matching entirely
)

// Template is a trained log template.
type Template struct {
	ID     int
	Tokens []string
	Count  int
}

// Matcher matches logs to trained Drain templates.
type Matcher struct {
	cfg         Config
	templates   []Template
	rootByLen   []*drainNode
	flatTree    drainFlatTree
	prefilter   matchPrefilter
	clusters    []*drainCluster
	dict        tokenDictionary
	paramID     uint64
	nextCluster int
	matchNeeded []int
	scratchIDs  []uint64
	scratchTok  []string
}

type drainNode struct {
	children   map[uint64]*drainNode
	clusterIDs []int
}

type drainCluster struct {
	id         int
	size       int
	paramCount int
	tokenIDs   []uint64
	tokenStr   []string
	// Pre-computed non-param indices for fast scoring.
	nonParamIdx []uint16
}

type drainFlatTree struct {
	rootByLen []uint32
	nodes     []drainFlatNode
}

type drainFlatNode struct {
	childKeys    []uint64
	childIndexes []uint32
	clusterIDs   []int
}

type prefilterEdgeKey struct {
	tokenCount int
	tokenID    uint64
}

type prefilterFirstLastKey struct {
	tokenCount int
	firstID    uint64
	lastID     uint64
}

type matchPrefilter struct {
	anyByLen       map[int][]int
	firstByEdge    map[prefilterEdgeKey][]int
	lastByEdge     map[prefilterEdgeKey][]int
	firstLastByKey map[prefilterFirstLastKey][]int

	// Compact index: per-length bucket for fast lookup.
	buckets []prefilterBucket
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

type tokenDictionary struct {
	ids    map[string]uint64
	nextID uint64
}

func newMatcher(cfg Config) *Matcher {
	m := &Matcher{
		cfg:         cfg,
		clusters:    make([]*drainCluster, 1),
		nextCluster: 1,
	}
	m.dict.init()
	m.paramID = m.dict.intern(cfg.ParamString)
	m.rebuildMatchNeeded()
	return m
}

func newDrainNode() *drainNode {
	return &drainNode{children: make(map[uint64]*drainNode)}
}

func newDrainCluster(id int, tokenStr []string, tokenIDs []uint64, size int, paramID uint64) *drainCluster {
	c := &drainCluster{id: id, size: size}
	c.setTemplate(tokenStr, tokenIDs, paramID)
	return c
}

func (c *drainCluster) setTemplate(tokenStr []string, tokenIDs []uint64, paramID uint64) {
	c.tokenStr = tokenStr
	c.tokenIDs = tokenIDs
	c.paramCount = 0
	// Pre-compute non-param indices for fast scoring loops.
	np := make([]uint16, 0, len(tokenIDs))
	for i := 0; i < len(tokenIDs); i++ {
		if tokenIDs[i] == paramID {
			c.paramCount++
		} else {
			np = append(np, uint16(i))
		}
	}
	c.nonParamIdx = np
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

// Match returns template id, extracted args, and whether a match was found.
func (m *Matcher) Match(line string) (templateID int, args []string, ok bool) {
	return m.MatchInto(line, nil)
}

// MatchInto returns template id, extracted args into dst, and whether a match was found.
func (m *Matcher) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	if m == nil {
		return 0, nil, false
	}
	var tokens []string
	if len(m.cfg.ExtraDelimiters) == 0 {
		tokenCount := 1
		for i := 0; i < len(line); i++ {
			if line[i] == ' ' {
				tokenCount++
				if tokenCount > maxMatchTokens {
					return 0, nil, false
				}
			}
		}
		if tokenCount >= len(m.flatTree.rootByLen) || m.flatTree.rootByLen[tokenCount] == 0 {
			return 0, nil, false
		}
		var tokenBuf [128]string
		tokens = tokenizeWhitespaceInto(line, tokenBuf[:0])
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
	}
	cluster := m.treeSearchDrainMatch(tokens, m.cfg.MatchThreshold)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, extractArgsByIDInto(cluster.tokenIDs, tokens, m.paramID, cluster.paramCount, dst), true
}

// MatchID returns template id and whether a match was found.
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	if m == nil {
		return 0, false
	}
	var tokens []string
	if len(m.cfg.ExtraDelimiters) == 0 {
		// Fast reject: count spaces to estimate token count and reject
		// lines that are too long or have no matching template length.
		tokenCount := 1
		for i := 0; i < len(line); i++ {
			if line[i] == ' ' {
				tokenCount++
				if tokenCount > maxMatchTokens {
					return 0, false
				}
			}
		}
		if tokenCount >= len(m.flatTree.rootByLen) || m.flatTree.rootByLen[tokenCount] == 0 {
			return 0, false
		}
		var tokenBuf [128]string
		tokens = tokenizeWhitespaceInto(line, tokenBuf[:0])
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
	}
	cluster := m.treeSearchDrainMatch(tokens, m.cfg.MatchThreshold)
	if cluster == nil {
		return 0, false
	}
	return cluster.id, true
}

func (m *Matcher) treeSearchDrainMatch(tokens []string, simTh float64) *drainCluster {
	tokenCount := len(tokens)

	// Fast length-based rejection: no templates exist with this token count.
	if tokenCount >= len(m.flatTree.rootByLen) || m.flatTree.rootByLen[tokenCount] == 0 {
		return nil
	}

	if m.cfg.EnableMatchPrefilter {
		var candidateBuf [64]int
		if candidateIDs, ok := m.prefilterCandidatesInto(tokens, candidateBuf[:0]); ok {
			return m.fastMatchDrainAdaptive(candidateIDs, tokens, simTh)
		}
	}

	curNodeIdx := m.flatRootNode(tokenCount)
	if curNodeIdx == 0 {
		return nil
	}
	curNode := &m.flatTree.nodes[curNodeIdx]

	if tokenCount == 0 {
		if len(curNode.clusterIDs) == 0 {
			return nil
		}
		return m.clusterByID(curNode.clusterIDs[0])
	}

	maxDepth := m.cfg.Depth - 2
	curDepth := 1
	for i := 0; i < tokenCount; i++ {
		if curDepth >= maxDepth || curDepth == tokenCount {
			break
		}

		nextNodeIdx := uint32(0)
		if tokenID, ok := m.dict.lookup(tokens[i]); ok {
			nextNodeIdx = flatNodeChild(curNode, tokenID)
		}
		if nextNodeIdx == 0 {
			nextNodeIdx = flatNodeChild(curNode, m.paramID)
		}
		if nextNodeIdx == 0 {
			return nil
		}

		curNode = &m.flatTree.nodes[nextNodeIdx]
		curDepth++
	}

	return m.fastMatchDrainAdaptive(curNode.clusterIDs, tokens, simTh)
}

// MarshalBinary serializes matcher as version + config + templates.
func (m *Matcher) MarshalBinary() ([]byte, error) {
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

// UnmarshalBinary deserializes matcher and rebuilds all runtime indexes.
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

	n, err := readUvarint(r)
	if err != nil {
		return fmt.Errorf("read template count: %w", err)
	}
	if n > uint64(^uint(0)>>1) {
		return errors.New("template count overflows int")
	}

	templates := make([]Template, int(n))
	for i := 0; i < len(templates); i++ {
		id, err := readVarint(r)
		if err != nil {
			return fmt.Errorf("read template id: %w", err)
		}
		count, err := readVarint(r)
		if err != nil {
			return fmt.Errorf("read template count: %w", err)
		}
		tokCount, err := readUvarint(r)
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

// LoadMatcher builds a matcher from a serialized payload.
func LoadMatcher(data []byte) (*Matcher, error) {
	var m Matcher
	if err := m.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	return &m, nil
}

func (m *Matcher) rebuildFromTemplates(cfg Config, templates []Template) error {
	norm, err := normalizeConfig(cfg)
	if err != nil {
		return err
	}

	next := newMatcher(norm)
	if len(templates) == 0 {
		*m = *next
		return nil
	}

	sorted := deepCopyTemplates(templates)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	seenIDs := make(map[int]struct{}, len(sorted))
	maxID := 0
	for i := range sorted {
		t := &sorted[i]
		if t.ID <= 0 {
			return fmt.Errorf("template id must be > 0, got %d", t.ID)
		}
		if _, exists := seenIDs[t.ID]; exists {
			return fmt.Errorf("duplicate template id %d", t.ID)
		}
		seenIDs[t.ID] = struct{}{}
		if t.Count <= 0 {
			return fmt.Errorf("template %d count must be > 0", t.ID)
		}
		if len(t.Tokens) > 0 {
			t.Tokens = copyStrings(t.Tokens)
		}
		if t.ID > maxID {
			maxID = t.ID
		}
	}

	next.clusters = make([]*drainCluster, maxID+1)
	next.nextCluster = maxID + 1

	for _, t := range sorted {
		ids := next.idsForTrain(t.Tokens)
		cluster := newDrainCluster(t.ID, copyStrings(t.Tokens), ids, t.Count, next.paramID)
		next.clusters[t.ID] = cluster
	}
	for id := 1; id < len(next.clusters); id++ {
		if c := next.clusters[id]; c != nil {
			next.addSeqToPrefixTreeDrain(c)
		}
	}
	next.syncTemplatesFromClusters()
	next.rebuildMatchPrefilter()
	next.rebuildFlatTree()
	next.rebuildMatchNeeded()
	if err := next.validateClusterReferences(); err != nil {
		return err
	}

	*m = *next
	return nil
}

func (m *Matcher) syncTemplatesFromClusters() {
	out := make([]Template, 0, len(m.clusters)-1)
	for id := 1; id < len(m.clusters); id++ {
		c := m.clusters[id]
		if c == nil {
			continue
		}
		out = append(out, Template{
			ID:     c.id,
			Tokens: c.tokenStr,
			Count:  c.size,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	m.templates = out
}

func (m *Matcher) idsForTrain(tokens []string) []uint64 {
	if len(tokens) == 0 {
		return nil
	}
	ids := make([]uint64, len(tokens))
	for i, tok := range tokens {
		ids[i] = m.dict.intern(tok)
	}
	return ids
}

func (m *Matcher) idsForTrainScratch(tokens []string) []uint64 {
	if len(tokens) == 0 {
		return nil
	}
	if cap(m.scratchIDs) < len(tokens) {
		m.scratchIDs = make([]uint64, len(tokens))
	} else {
		m.scratchIDs = m.scratchIDs[:len(tokens)]
	}
	for i, tok := range tokens {
		m.scratchIDs[i] = m.dict.intern(tok)
	}
	return m.scratchIDs
}

func (d *tokenDictionary) init() {
	d.ids = make(map[string]uint64)
	d.nextID = 1
}

func (d *tokenDictionary) intern(token string) uint64 {
	if id, ok := d.ids[token]; ok {
		return id
	}
	id := d.nextID
	d.nextID++
	d.ids[token] = id
	return id
}

func (d *tokenDictionary) lookup(token string) (uint64, bool) {
	id, ok := d.ids[token]
	return id, ok
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

func extractArgsByID(templateTokenIDs []uint64, lineTokens []string, paramID uint64, paramCount int) []string {
	return extractArgsByIDInto(templateTokenIDs, lineTokens, paramID, paramCount, nil)
}

func extractArgsByIDInto(templateTokenIDs []uint64, lineTokens []string, paramID uint64, paramCount int, dst []string) []string {
	if len(templateTokenIDs) == 0 || len(lineTokens) == 0 || paramCount == 0 {
		return nil
	}
	limit := len(templateTokenIDs)
	if len(lineTokens) < limit {
		limit = len(lineTokens)
	}
	if paramCount > limit {
		paramCount = limit
	}
	args := dst[:0]
	if cap(args) < paramCount {
		args = make([]string, 0, paramCount)
	}
	for i := 0; i < limit; i++ {
		if templateTokenIDs[i] == paramID {
			args = append(args, lineTokens[i])
		}
	}
	if len(args) == 0 {
		return nil
	}
	return args
}

func hasNumbers(s string) bool {
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch >= '0' && ch <= '9' {
			return true
		}
	}
	return false
}

func copyStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	copy(out, in)
	return out
}

func deepCopyTemplates(in []Template) []Template {
	if len(in) == 0 {
		return nil
	}
	out := make([]Template, len(in))
	for i := range in {
		out[i] = Template{
			ID:     in[i].ID,
			Tokens: copyStrings(in[i].Tokens),
			Count:  in[i].Count,
		}
	}
	return out
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

func readConfigBinary(r *bytes.Reader) (Config, error) {
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
	n, err := readUvarint(r)
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

func readUvarint(r *bytes.Reader) (uint64, error) {
	v, err := binary.ReadUvarint(r)
	if err != nil {
		return 0, err
	}
	return v, nil
}

func writeVarint(w *bytes.Buffer, v int64) error {
	var scratch [binary.MaxVarintLen64]byte
	n := binary.PutVarint(scratch[:], v)
	_, err := w.Write(scratch[:n])
	return err
}

func readVarint(r *bytes.Reader) (int64, error) {
	v, err := binary.ReadVarint(r)
	if err != nil {
		return 0, err
	}
	return v, nil
}

// Train trains a matcher with default config.
func Train(samples []string) (*Matcher, error) {
	return TrainWithConfig(samples, DefaultConfig())
}

// TrainWithConfig trains a matcher with custom config.
func TrainWithConfig(samples []string, cfg Config) (*Matcher, error) {
	norm, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}

	m := newMatcher(norm)
	for _, sample := range samples {
		m.addLogMessage(sample)
	}
	m.syncTemplatesFromClusters()
	m.rebuildMatchPrefilter()
	m.rebuildFlatTree()
	m.rebuildMatchNeeded()
	if err := m.validateClusterReferences(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *Matcher) addLogMessage(content string) {
	tokens := m.tokenizeForTrain(content)
	if len(tokens) > maxMatchTokens {
		return // skip oversized lines — they can never be matched
	}
	tokenIDs := m.idsForTrainScratch(tokens)

	matchCluster := m.treeSearchDrain(tokenIDs, m.cfg.SimilarityThreshold, false)
	if matchCluster == nil {
		clusterID := m.nextCluster
		m.nextCluster++
		cluster := newDrainCluster(clusterID, copyStrings(tokens), copyUint64s(tokenIDs), 1, m.paramID)
		m.ensureClusterSize(clusterID)
		m.clusters[clusterID] = cluster
		m.addSeqToPrefixTreeDrain(cluster)
		return
	}

	for i := 0; i < len(tokenIDs) && i < len(matchCluster.tokenIDs); i++ {
		if matchCluster.tokenIDs[i] == m.paramID {
			continue
		}
		if matchCluster.tokenIDs[i] != tokenIDs[i] {
			matchCluster.tokenIDs[i] = m.paramID
			matchCluster.tokenStr[i] = m.cfg.ParamString
			matchCluster.paramCount++
		}
	}
	matchCluster.size++
}

func (m *Matcher) tokenizeForTrain(content string) []string {
	if len(m.cfg.ExtraDelimiters) > 0 {
		return tokenize(content, m.cfg.ExtraDelimiters)
	}
	m.scratchTok = tokenizeWhitespaceInto(content, m.scratchTok[:0])
	return m.scratchTok
}

func (m *Matcher) ensureClusterSize(id int) {
	if id < len(m.clusters) {
		return
	}
	need := id - len(m.clusters) + 1
	m.clusters = append(m.clusters, make([]*drainCluster, need)...)
}

func copyUint64s(in []uint64) []uint64 {
	if len(in) == 0 {
		return nil
	}
	out := make([]uint64, len(in))
	copy(out, in)
	return out
}

func (m *Matcher) clusterByID(id int) *drainCluster {
	if id <= 0 || id >= len(m.clusters) {
		return nil
	}
	return m.clusters[id]
}

func (m *Matcher) treeSearchDrain(tokenIDs []uint64, simTh float64, includeParams bool) *drainCluster {
	tokenCount := len(tokenIDs)
	curNode := m.mutableRootNode(tokenCount)
	if curNode == nil {
		return nil
	}

	if tokenCount == 0 {
		if len(curNode.clusterIDs) == 0 {
			return nil
		}
		return m.clusterByID(curNode.clusterIDs[0])
	}

	maxDepth := m.cfg.Depth - 2
	curDepth := 1
	for i := 0; i < tokenCount; i++ {
		if curDepth >= maxDepth {
			break
		}
		if curDepth == tokenCount {
			break
		}

		tokenID := tokenIDs[i]
		nextNode := curNode.children[tokenID]
		if nextNode == nil {
			nextNode = curNode.children[m.paramID]
		}
		if nextNode == nil {
			return nil
		}

		curNode = nextNode
		curDepth++
	}

	return m.fastMatchDrain(curNode.clusterIDs, tokenIDs, simTh, includeParams)
}

func (m *Matcher) fastMatchDrain(clusterIDs []int, tokenIDs []uint64, simTh float64, includeParams bool) *drainCluster {
	nTokens := len(tokenIDs)
	requiredScore := m.requiredScore(nTokens, simTh)
	maxScore := -1
	maxParamCount := -1
	var maxCluster *drainCluster
	clusters := m.clusters

	for _, clusterID := range clusterIDs {
		cluster := clusters[clusterID]
		cIDs := cluster.tokenIDs

		// Quick length check.
		if len(cIDs) != nTokens {
			continue
		}

		// Inline scoring: iterate non-param indices, compare uint64 IDs.
		paramCount := cluster.paramCount
		simTokens := 0
		if includeParams {
			simTokens = paramCount
		}
		npIdx := cluster.nonParamIdx
		remaining := len(npIdx)
		for _, idx := range npIdx {
			if cIDs[idx] == tokenIDs[idx] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < requiredScore {
				break
			}
		}

		if simTokens > maxScore || (simTokens == maxScore && paramCount > maxParamCount) {
			maxScore = simTokens
			maxParamCount = paramCount
			maxCluster = cluster
		}
	}

	if maxScore >= requiredScore {
		return maxCluster
	}
	return nil
}


func (m *Matcher) fastMatchDrainAdaptive(clusterIDs []int, tokens []string, simTh float64) *drainCluster {
	if len(clusterIDs) == 0 {
		return nil
	}
	// Always resolve to IDs. Unknown tokens get sentinel value 0 and are
	// counted as automatic mismatches (they can't match any dictionary token).
	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.fastMatchDrain(clusterIDs, tokenIDs, simTh, true)
}

// lookupTokenIDsPartial resolves tokens to dictionary IDs. Unknown tokens
// are set to 0 (which never matches any valid token ID, since IDs start at 1).
func (m *Matcher) lookupTokenIDsPartial(tokens []string, dst []uint64) []uint64 {
	if len(tokens) == 0 {
		return nil
	}
	if cap(dst) < len(tokens) {
		dst = make([]uint64, len(tokens))
	} else {
		dst = dst[:len(tokens)]
	}
	for i := range tokens {
		dst[i], _ = m.dict.lookup(tokens[i]) // unknown tokens get 0, guaranteed mismatch
	}
	return dst
}



func (m *Matcher) addSeqToPrefixTreeDrain(cluster *drainCluster) {
	tokenCount := len(cluster.tokenIDs)
	firstLayerNode := m.mutableRootNode(tokenCount)
	if firstLayerNode == nil {
		firstLayerNode = newDrainNode()
		m.setMutableRootNode(tokenCount, firstLayerNode)
	}

	curNode := firstLayerNode
	if tokenCount == 0 {
		curNode.clusterIDs = []int{cluster.id}
		return
	}

	curDepth := 1
	for i, tokenID := range cluster.tokenIDs {
		if curDepth >= m.cfg.Depth-2 || curDepth >= tokenCount {
			curNode.clusterIDs = append(curNode.clusterIDs, cluster.id)
			break
		}

		nextNode := curNode.children[tokenID]
		if nextNode == nil {
			if m.cfg.ParametrizeNumericTokens && hasNumbers(cluster.tokenStr[i]) {
				nextNode = curNode.children[m.paramID]
				if nextNode == nil {
					nextNode = newDrainNode()
					curNode.children[m.paramID] = nextNode
				}
			} else {
				wildcardNode := curNode.children[m.paramID]
				if wildcardNode != nil {
					if len(curNode.children) < m.cfg.MaxChildren {
						nextNode = newDrainNode()
						curNode.children[tokenID] = nextNode
					} else {
						nextNode = wildcardNode
					}
				} else {
					nextChildren := len(curNode.children) + 1
					if nextChildren < m.cfg.MaxChildren {
						nextNode = newDrainNode()
						curNode.children[tokenID] = nextNode
					} else if nextChildren == m.cfg.MaxChildren {
						nextNode = newDrainNode()
						curNode.children[m.paramID] = nextNode
					} else {
						nextNode = curNode.children[m.paramID]
						if nextNode == nil {
							nextNode = newDrainNode()
							curNode.children[m.paramID] = nextNode
						}
					}
				}
			}
		}

		curNode = nextNode
		curDepth++
	}
}

func (m *Matcher) mutableRootNode(tokenCount int) *drainNode {
	if tokenCount < 0 || tokenCount >= len(m.rootByLen) {
		return nil
	}
	return m.rootByLen[tokenCount]
}

func (m *Matcher) setMutableRootNode(tokenCount int, node *drainNode) {
	if tokenCount < 0 {
		return
	}
	if tokenCount >= len(m.rootByLen) {
		m.rootByLen = append(m.rootByLen, make([]*drainNode, tokenCount-len(m.rootByLen)+1)...)
	}
	m.rootByLen[tokenCount] = node
}

func (m *Matcher) flatRootNode(tokenCount int) uint32 {
	if tokenCount < 0 || tokenCount >= len(m.flatTree.rootByLen) {
		return 0
	}
	return m.flatTree.rootByLen[tokenCount]
}

func (m *Matcher) rebuildFlatTree() {
	m.flatTree.rootByLen = make([]uint32, len(m.rootByLen))
	m.flatTree.nodes = make([]drainFlatNode, 1) // index 0 means "missing node"
	if len(m.rootByLen) == 0 {
		return
	}

	seen := make(map[*drainNode]uint32, len(m.rootByLen))
	for tokenCount, root := range m.rootByLen {
		if root == nil {
			continue
		}
		m.flatTree.rootByLen[tokenCount] = flattenDrainNode(root, &m.flatTree, seen)
	}
}

func flattenDrainNode(node *drainNode, tree *drainFlatTree, seen map[*drainNode]uint32) uint32 {
	if idx, ok := seen[node]; ok {
		return idx
	}

	idx := uint32(len(tree.nodes))
	seen[node] = idx
	tree.nodes = append(tree.nodes, drainFlatNode{
		clusterIDs: append([]int(nil), node.clusterIDs...),
	})

	if len(node.children) == 0 {
		return idx
	}

	keys := make([]uint64, 0, len(node.children))
	for key := range node.children {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	children := make([]uint32, len(keys))
	for i, key := range keys {
		children[i] = flattenDrainNode(node.children[key], tree, seen)
	}

	flat := &tree.nodes[idx]
	flat.childKeys = keys
	flat.childIndexes = children
	return idx
}

func flatNodeChild(node *drainFlatNode, tokenID uint64) uint32 {
	if len(node.childKeys) <= 8 {
		for i, key := range node.childKeys {
			if key == tokenID {
				return node.childIndexes[i]
			}
		}
		return 0
	}

	lo, hi := 0, len(node.childKeys)
	for lo < hi {
		mid := lo + (hi-lo)/2
		key := node.childKeys[mid]
		if key == tokenID {
			return node.childIndexes[mid]
		}
		if key < tokenID {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return 0
}

func (m *Matcher) rebuildMatchNeeded() {
	m.matchNeeded = make([]int, len(m.rootByLen))
	for tokenCount := range len(m.matchNeeded) {
		m.matchNeeded[tokenCount] = int(math.Ceil(m.cfg.MatchThreshold * float64(tokenCount)))
	}
}

func (m *Matcher) requiredScore(tokenCount int, simTh float64) int {
	if simTh == m.cfg.MatchThreshold && tokenCount >= 0 && tokenCount < len(m.matchNeeded) {
		return m.matchNeeded[tokenCount]
	}
	return int(math.Ceil(simTh * float64(tokenCount)))
}

func (m *Matcher) rebuildMatchPrefilter() {
	if !m.cfg.EnableMatchPrefilter {
		m.prefilter = matchPrefilter{}
		return
	}

	pf := matchPrefilter{
		anyByLen:       make(map[int][]int),
		firstByEdge:    make(map[prefilterEdgeKey][]int),
		lastByEdge:     make(map[prefilterEdgeKey][]int),
		firstLastByKey: make(map[prefilterFirstLastKey][]int),
	}

	maxLen := 0
	for id := 1; id < len(m.clusters); id++ {
		cluster := m.clusters[id]
		if cluster == nil {
			continue
		}

		tokenCount := len(cluster.tokenIDs)
		if tokenCount > maxLen {
			maxLen = tokenCount
		}
		if tokenCount == 0 {
			pf.anyByLen[0] = append(pf.anyByLen[0], id)
			continue
		}

		firstID := cluster.tokenIDs[0]
		lastID := cluster.tokenIDs[tokenCount-1]
		firstIsParam := firstID == m.paramID
		lastIsParam := lastID == m.paramID

		switch {
		case firstIsParam && lastIsParam:
			pf.anyByLen[tokenCount] = append(pf.anyByLen[tokenCount], id)
		case !firstIsParam && lastIsParam:
			key := prefilterEdgeKey{tokenCount: tokenCount, tokenID: firstID}
			pf.firstByEdge[key] = append(pf.firstByEdge[key], id)
		case firstIsParam && !lastIsParam:
			key := prefilterEdgeKey{tokenCount: tokenCount, tokenID: lastID}
			pf.lastByEdge[key] = append(pf.lastByEdge[key], id)
		default:
			key := prefilterFirstLastKey{tokenCount: tokenCount, firstID: firstID, lastID: lastID}
			pf.firstLastByKey[key] = append(pf.firstLastByKey[key], id)
		}
	}

	// Build compact buckets from maps for faster lookup.
	pf.buckets = make([]prefilterBucket, maxLen+1)
	for tc, ids := range pf.anyByLen {
		if tc < len(pf.buckets) {
			pf.buckets[tc].any = ids
		}
	}
	// Build sorted first-edge arrays per token count.
	firstByTC := make(map[int]map[uint64][]int)
	for key, ids := range pf.firstByEdge {
		if firstByTC[key.tokenCount] == nil {
			firstByTC[key.tokenCount] = make(map[uint64][]int)
		}
		firstByTC[key.tokenCount][key.tokenID] = ids
	}
	for tc, m := range firstByTC {
		if tc >= len(pf.buckets) {
			continue
		}
		b := &pf.buckets[tc]
		b.firstKeys = make([]uint64, 0, len(m))
		for k := range m {
			b.firstKeys = append(b.firstKeys, k)
		}
		sort.Slice(b.firstKeys, func(i, j int) bool { return b.firstKeys[i] < b.firstKeys[j] })
		b.firstVals = make([][]int, len(b.firstKeys))
		for i, k := range b.firstKeys {
			b.firstVals[i] = m[k]
		}
	}
	// Build sorted last-edge arrays per token count.
	lastByTC := make(map[int]map[uint64][]int)
	for key, ids := range pf.lastByEdge {
		if lastByTC[key.tokenCount] == nil {
			lastByTC[key.tokenCount] = make(map[uint64][]int)
		}
		lastByTC[key.tokenCount][key.tokenID] = ids
	}
	for tc, m := range lastByTC {
		if tc >= len(pf.buckets) {
			continue
		}
		b := &pf.buckets[tc]
		b.lastKeys = make([]uint64, 0, len(m))
		for k := range m {
			b.lastKeys = append(b.lastKeys, k)
		}
		sort.Slice(b.lastKeys, func(i, j int) bool { return b.lastKeys[i] < b.lastKeys[j] })
		b.lastVals = make([][]int, len(b.lastKeys))
		for i, k := range b.lastKeys {
			b.lastVals[i] = m[k]
		}
	}
	// Build sorted firstLast arrays per token count.
	flByTC := make(map[int]map[uint64][]int)
	for key, ids := range pf.firstLastByKey {
		if flByTC[key.tokenCount] == nil {
			flByTC[key.tokenCount] = make(map[uint64][]int)
		}
		combined := (key.firstID << 32) | (key.lastID & 0xFFFFFFFF)
		flByTC[key.tokenCount][combined] = ids
	}
	for tc, m := range flByTC {
		if tc >= len(pf.buckets) {
			continue
		}
		b := &pf.buckets[tc]
		b.flKeys = make([]uint64, 0, len(m))
		for k := range m {
			b.flKeys = append(b.flKeys, k)
		}
		sort.Slice(b.flKeys, func(i, j int) bool { return b.flKeys[i] < b.flKeys[j] })
		b.flVals = make([][]int, len(b.flKeys))
		for i, k := range b.flKeys {
			b.flVals[i] = m[k]
		}
	}

	m.prefilter = pf
}

func (m *Matcher) prefilterCandidatesInto(tokens []string, dst []int) ([]int, bool) {
	tokenCount := len(tokens)

	// Use compact buckets if available.
	if tokenCount < len(m.prefilter.buckets) {
		return m.prefilterCandidatesCompact(tokens, tokenCount, dst)
	}

	// Fallback to map-based lookup for out-of-range token counts.
	any := m.prefilter.anyByLen[tokenCount]
	var first []int
	var last []int
	var firstLast []int
	if tokenCount > 0 {
		firstID, firstKnown := m.dict.lookup(tokens[0])
		lastID, lastKnown := m.dict.lookup(tokens[tokenCount-1])
		if firstKnown {
			first = m.prefilter.firstByEdge[prefilterEdgeKey{
				tokenCount: tokenCount,
				tokenID:    firstID,
			}]
		}
		if lastKnown {
			last = m.prefilter.lastByEdge[prefilterEdgeKey{
				tokenCount: tokenCount,
				tokenID:    lastID,
			}]
		}
		if firstKnown && lastKnown {
			firstLast = m.prefilter.firstLastByKey[prefilterFirstLastKey{
				tokenCount: tokenCount,
				firstID:    firstID,
				lastID:     lastID,
			}]
		}
	}
	return mergePrefilterGroups(any, first, last, firstLast, dst)
}

func (m *Matcher) prefilterCandidatesCompact(tokens []string, tokenCount int, dst []int) ([]int, bool) {
	b := &m.prefilter.buckets[tokenCount]
	any := b.any

	var first []int
	var last []int
	var firstLast []int
	if tokenCount > 0 {
		firstID, firstKnown := m.dict.lookup(tokens[0])
		lastID, lastKnown := m.dict.lookup(tokens[tokenCount-1])
		if firstKnown {
			first = searchSortedU64(b.firstKeys, b.firstVals, firstID)
		}
		if lastKnown {
			last = searchSortedU64(b.lastKeys, b.lastVals, lastID)
		}
		if firstKnown && lastKnown {
			combined := (firstID << 32) | (lastID & 0xFFFFFFFF)
			firstLast = searchSortedU64(b.flKeys, b.flVals, combined)
		}
	}
	return mergePrefilterGroups(any, first, last, firstLast, dst)
}

func searchSortedU64(keys []uint64, vals [][]int, target uint64) []int {
	lo, hi := 0, len(keys)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if keys[mid] == target {
			return vals[mid]
		}
		if keys[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return nil
}

func mergePrefilterGroups(any, first, last, firstLast []int, dst []int) ([]int, bool) {
	nonEmpty := 0
	var single []int
	total := 0
	for _, group := range [...][]int{any, first, last, firstLast} {
		if len(group) == 0 {
			continue
		}
		nonEmpty++
		single = group
		total += len(group)
	}
	if nonEmpty == 0 {
		return nil, false
	}
	if nonEmpty == 1 {
		return single, true
	}

	out := dst[:0]
	if cap(out) < total {
		out = make([]int, 0, total)
	}
	out = append(out, any...)
	out = append(out, first...)
	out = append(out, last...)
	out = append(out, firstLast...)
	return out, true
}

func (m *Matcher) validateClusterReferences() error {
	seen := make(map[*drainNode]struct{})
	stack := make([]*drainNode, 0, len(m.rootByLen))
	for _, root := range m.rootByLen {
		if root != nil {
			stack = append(stack, root)
		}
	}

	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if _, ok := seen[node]; ok {
			continue
		}
		seen[node] = struct{}{}

		for _, clusterID := range node.clusterIDs {
			if clusterID <= 0 || clusterID >= len(m.clusters) {
				return fmt.Errorf("invalid cluster id %d in tree", clusterID)
			}
			if m.clusters[clusterID] == nil {
				return fmt.Errorf("nil cluster at id %d in tree", clusterID)
			}
		}
		for _, child := range node.children {
			if child != nil {
				stack = append(stack, child)
			}
		}
	}

	if !m.cfg.EnableMatchPrefilter {
		return nil
	}

	for _, ids := range m.prefilter.anyByLen {
		if err := m.validateClusterIDSlice(ids); err != nil {
			return err
		}
	}
	for _, ids := range m.prefilter.firstByEdge {
		if err := m.validateClusterIDSlice(ids); err != nil {
			return err
		}
	}
	for _, ids := range m.prefilter.lastByEdge {
		if err := m.validateClusterIDSlice(ids); err != nil {
			return err
		}
	}
	for _, ids := range m.prefilter.firstLastByKey {
		if err := m.validateClusterIDSlice(ids); err != nil {
			return err
		}
	}
	return nil
}

func (m *Matcher) validateClusterIDSlice(clusterIDs []int) error {
	for _, clusterID := range clusterIDs {
		if clusterID <= 0 || clusterID >= len(m.clusters) {
			return fmt.Errorf("invalid cluster id %d", clusterID)
		}
		if m.clusters[clusterID] == nil {
			return fmt.Errorf("nil cluster at id %d", clusterID)
		}
	}
	return nil
}
