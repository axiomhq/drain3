package drain3

import (
	"cmp"
	"fmt"
	"slices"
)

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

// NewMatcherFromTemplates builds a matcher from pre-existing templates.
func NewMatcherFromTemplates(cfg Config, templates []Template) (*Matcher, error) {
	var m Matcher
	if err := m.rebuildFromTemplates(cfg, templates); err != nil {
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

	sorted := slices.Clone(templates)
	slices.SortFunc(sorted, func(a, b Template) int { return cmp.Compare(a.ID, b.ID) })

	seenIDs := make(map[int]struct{}, len(sorted))
	maxID := 0
	for _, t := range sorted {
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
		if t.ID > maxID {
			maxID = t.ID
		}
	}

	next.clusters = make([]*cluser, maxID+1)
	next.nextCluster = maxID + 1

	for _, t := range sorted {
		ids := next.internTokenIDs(t.Tokens, nil)
		cluster := newCluster(t.ID, slices.Clone(t.Tokens), ids, t.Count, next.paramID)
		next.clusters[t.ID] = cluster
	}
	for id := 1; id < len(next.clusters); id++ {
		if c := next.clusters[id]; c != nil {
			next.addSeqToPrefixTree(c)
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
	slices.SortFunc(out, func(a, b Template) int { return cmp.Compare(b.Count, a.Count) })
	m.templates = out
}

func (m *Matcher) addLogMessage(content string) {
	if len(content) > m.cfg.MaxBytes {
		return // skip oversized lines — they can never be matched
	}
	var tokens []string
	if len(m.cfg.ExtraDelimiters) > 0 {
		tokens = tokenize(content, m.cfg.ExtraDelimiters)
	} else {
		m.scratchTok = tokenizeWhitespaceInto(content, m.scratchTok[:0])
		tokens = m.scratchTok
	}
	if len(tokens) > m.cfg.MaxTokens {
		return
	}
	m.scratchIDs = m.internTokenIDs(tokens, m.scratchIDs)
	tokenIDs := m.scratchIDs

	matchCluster := m.treeSearch(tokenIDs, m.cfg.SimilarityThreshold)
	if matchCluster == nil {
		clusterID := m.nextCluster
		m.nextCluster++
		cluster := newCluster(clusterID, slices.Clone(tokens), slices.Clone(tokenIDs), 1, m.paramID)
		if clusterID >= len(m.clusters) {
			m.clusters = append(m.clusters, make([]*cluser, clusterID-len(m.clusters)+1)...)
		}
		m.clusters[clusterID] = cluster
		m.addSeqToPrefixTree(cluster)
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

func (m *Matcher) treeSearch(tokenIDs []uint64, simTh float64) *cluser {
	tokenCount := len(tokenIDs)
	if tokenCount >= len(m.rootByLen) {
		return nil
	}
	curNode := m.rootByLen[tokenCount]
	if curNode == nil {
		return nil
	}

	if tokenCount == 0 {
		if len(curNode.clusterIDs) == 0 {
			return nil
		}
		return m.clusters[curNode.clusterIDs[0]]
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

	return m.fastMatch(curNode.clusterIDs, tokenIDs, simTh, false)
}

func (m *Matcher) addSeqToPrefixTree(cluster *cluser) {
	tokenCount := len(cluster.tokenIDs)
	if tokenCount >= len(m.rootByLen) {
		m.rootByLen = append(m.rootByLen, make([]*Node, tokenCount-len(m.rootByLen)+1)...)
	}
	firstLayerNode := m.rootByLen[tokenCount]
	if firstLayerNode == nil {
		firstLayerNode = newNode()
		m.rootByLen[tokenCount] = firstLayerNode
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
					nextNode = newNode()
					curNode.children[m.paramID] = nextNode
				}
			} else {
				wildcardNode := curNode.children[m.paramID]
				if wildcardNode != nil {
					if len(curNode.children) < m.cfg.MaxChildren {
						nextNode = newNode()
						curNode.children[tokenID] = nextNode
					} else {
						nextNode = wildcardNode
					}
				} else {
					nextChildren := len(curNode.children) + 1
					if nextChildren < m.cfg.MaxChildren {
						nextNode = newNode()
						curNode.children[tokenID] = nextNode
					} else if nextChildren == m.cfg.MaxChildren {
						nextNode = newNode()
						curNode.children[m.paramID] = nextNode
					} else {
						nextNode = curNode.children[m.paramID]
						if nextNode == nil {
							nextNode = newNode()
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

func (m *Matcher) validateClusterReferences() error {
	seen := make(map[*Node]struct{})
	stack := make([]*Node, 0, len(m.rootByLen))
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

	for _, b := range m.prefilterBuckets {
		for _, groups := range [...][][]int{
			{b.any},
			b.firstVals,
			b.lastVals,
			b.flVals,
		} {
			for _, ids := range groups {
				for _, clusterID := range ids {
					if clusterID <= 0 || clusterID >= len(m.clusters) {
						return fmt.Errorf("invalid cluster id %d", clusterID)
					}
					if m.clusters[clusterID] == nil {
						return fmt.Errorf("nil cluster at id %d", clusterID)
					}
				}
			}
		}
	}
	return nil
}
