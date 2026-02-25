package drain3

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

const (
	payloadMagic                = "drn3"
	payloadVersion              = byte(1)
	adaptiveIDPathMinCandidates = 16
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
	root        map[int]*drainNode
	clusters    []*drainCluster
	dict        tokenDictionary
	paramID     uint64
	nextCluster int
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
}

type tokenDictionary struct {
	ids    map[string]uint64
	nextID uint64
}

func newMatcher(cfg Config) *Matcher {
	m := &Matcher{
		cfg:         cfg,
		root:        make(map[int]*drainNode),
		clusters:    make([]*drainCluster, 1),
		nextCluster: 1,
	}
	m.dict.init()
	m.paramID = m.dict.intern(cfg.ParamString)
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
	for i := 0; i < len(tokenIDs); i++ {
		if tokenIDs[i] == paramID {
			c.paramCount++
		}
	}
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
		var tokenBuf [32]string
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
		var tokenBuf [32]string
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
	curNode := m.root[tokenCount]
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
		if curDepth >= maxDepth || curDepth == tokenCount {
			break
		}

		nextNode := (*drainNode)(nil)
		if tokenID, ok := m.dict.lookup(tokens[i]); ok {
			nextNode = curNode.children[tokenID]
		}
		if nextNode == nil {
			nextNode = curNode.children[m.paramID]
		}
		if nextNode == nil {
			return nil
		}

		curNode = nextNode
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
	return m, nil
}

func (m *Matcher) addLogMessage(content string) {
	tokens := m.tokenizeForTrain(content)
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
	curNode := m.root[tokenCount]
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
	requiredScore := simTh * float64(len(tokenIDs))
	maxScore := -1
	maxParamCount := -1
	var maxCluster *drainCluster

	for _, clusterID := range clusterIDs {
		cluster := m.clusterByID(clusterID)
		if cluster == nil {
			continue
		}
		curScore, paramCount := m.seqScoreDrain(cluster.tokenIDs, cluster.paramCount, tokenIDs, includeParams)
		if curScore > maxScore || (curScore == maxScore && paramCount > maxParamCount) {
			maxScore = curScore
			maxParamCount = paramCount
			maxCluster = cluster
		}
	}

	if float64(maxScore) >= requiredScore {
		return maxCluster
	}
	return nil
}

func (m *Matcher) fastMatchDrainTokens(clusterIDs []int, tokens []string, simTh float64) *drainCluster {
	requiredScore := simTh * float64(len(tokens))
	maxScore := -1
	maxParamCount := -1
	var maxCluster *drainCluster

	for _, clusterID := range clusterIDs {
		cluster := m.clusterByID(clusterID)
		if cluster == nil {
			continue
		}
		curScore, paramCount := m.seqScoreDrainTokens(cluster, tokens)
		if curScore > maxScore || (curScore == maxScore && paramCount > maxParamCount) {
			maxScore = curScore
			maxParamCount = paramCount
			maxCluster = cluster
		}
	}

	if float64(maxScore) >= requiredScore {
		return maxCluster
	}
	return nil
}

func (m *Matcher) fastMatchDrainAdaptive(clusterIDs []int, tokens []string, simTh float64) *drainCluster {
	if len(clusterIDs) >= adaptiveIDPathMinCandidates {
		var tokenIDBuf [32]uint64
		if tokenIDs, ok := m.lookupKnownTokenIDs(tokens, tokenIDBuf[:0]); ok {
			return m.fastMatchDrain(clusterIDs, tokenIDs, simTh, true)
		}
	}
	return m.fastMatchDrainTokens(clusterIDs, tokens, simTh)
}

func (m *Matcher) lookupKnownTokenIDs(tokens []string, dst []uint64) ([]uint64, bool) {
	if len(tokens) == 0 {
		return nil, true
	}
	if cap(dst) < len(tokens) {
		dst = make([]uint64, len(tokens))
	} else {
		dst = dst[:len(tokens)]
	}
	for i := range len(tokens) {
		tokenID, ok := m.dict.lookup(tokens[i])
		if !ok {
			return nil, false
		}
		dst[i] = tokenID
	}
	return dst, true
}

func (m *Matcher) seqScoreDrain(seq1 []uint64, paramCount int, seq2 []uint64, includeParams bool) (int, int) {
	if len(seq1) != len(seq2) {
		return 0, 0
	}
	if len(seq1) == 0 {
		return 1, 0
	}

	simTokens := 0
	for i := 0; i < len(seq1); i++ {
		t1 := seq1[i]
		if t1 == m.paramID {
			continue
		}
		if t1 == seq2[i] {
			simTokens++
		}
	}
	if includeParams {
		simTokens += paramCount
	}

	return simTokens, paramCount
}

func (m *Matcher) seqScoreDrainTokens(cluster *drainCluster, tokens []string) (int, int) {
	if len(cluster.tokenStr) != len(tokens) {
		return 0, 0
	}
	if len(tokens) == 0 {
		return 1, 0
	}

	simTokens := cluster.paramCount
	for i := 0; i < len(cluster.tokenIDs); i++ {
		if cluster.tokenIDs[i] == m.paramID {
			continue
		}
		if cluster.tokenStr[i] == tokens[i] {
			simTokens++
		}
	}
	return simTokens, cluster.paramCount
}

func (m *Matcher) addSeqToPrefixTreeDrain(cluster *drainCluster) {
	tokenCount := len(cluster.tokenIDs)
	firstLayerNode := m.root[tokenCount]
	if firstLayerNode == nil {
		firstLayerNode = newDrainNode()
		m.root[tokenCount] = firstLayerNode
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
