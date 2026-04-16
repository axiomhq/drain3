package drain3

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/bits-and-blooms/bitset"
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
	m.rebuildMatchNeeded()
	m.freezeDict()
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

	next.clusters = make([]*cluster, maxID+1)
	next.nextCluster = maxID + 1

	for _, t := range sorted {
		full := make([]string, t.TokenCount)
		denseIdx := 0
		for i := 0; i < t.TokenCount; i++ {
			if t.Params.Test(uint(i)) {
				full[i] = next.cfg.ParamString
			} else {
				full[i] = t.Tokens[denseIdx]
				denseIdx++
			}
		}
		ids := next.internTokenIDs(full, nil)
		cluster := newCluster(t.ID, full, ids, t.Count, next.paramID)
		next.clusters[t.ID] = cluster
	}
	for id := 1; id < len(next.clusters); id++ {
		if c := next.clusters[id]; c != nil {
			next.addSeqToPrefixTree(c)
		}
	}
	next.syncTemplatesFromClusters()
	next.rebuildMatchPrefilter()
	next.rebuildMatchNeeded()
	next.freezeDict()

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
		tokenCount := len(c.tokenIDs)
		params := bitset.New(uint(tokenCount))
		dense := make([]string, 0, tokenCount-c.paramCount)
		for i, tid := range c.tokenIDs {
			if tid == m.paramID {
				params.Set(uint(i))
			} else {
				dense = append(dense, c.tokenStr[i])
			}
		}
		out = append(out, Template{
			ID:         c.id,
			Tokens:     dense,
			Params:     params,
			TokenCount: tokenCount,
			Count:      c.size,
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
		if len(tokens) > m.cfg.MaxTokens {
			return
		}
	} else {
		var tokenCount int
		m.scratchTok, tokenCount = tokenizeWhitespaceCount(content, m.scratchTok, m.cfg.MaxTokens)
		if tokenCount > m.cfg.MaxTokens {
			return
		}
		tokens = m.scratchTok
	}

	matchCluster := m.treeSearchStrings(tokens, m.cfg.SimilarityThreshold, false)
	if matchCluster == nil {
		if m.cfg.MaxClusters > 0 && m.nextCluster-1 >= m.cfg.MaxClusters {
			return // cluster cap reached, drop line
		}
		m.scratchIDs = m.internTokenIDs(tokens, m.scratchIDs)
		tokenIDs := m.scratchIDs
		clusterID := m.nextCluster
		m.nextCluster++
		cl := newCluster(clusterID, slices.Clone(tokens), slices.Clone(tokenIDs), 1, m.paramID)
		if clusterID >= len(m.clusters) {
			m.clusters = append(m.clusters, make([]*cluster, clusterID-len(m.clusters)+1)...)
		}
		m.clusters[clusterID] = cl
		m.addSeqToPrefixTree(cl)
		return
	}

	changed := false
	for i := 0; i < len(tokens) && i < len(matchCluster.tokenStr); i++ {
		if matchCluster.tokenIDs[i] == m.paramID {
			continue
		}
		if matchCluster.tokenStr[i] != tokens[i] {
			matchCluster.tokenIDs[i] = m.paramID
			matchCluster.tokenStr[i] = m.cfg.ParamString
			matchCluster.paramCount++
			changed = true
		}
	}
	if changed {
		matchCluster.rebuildNonParamIdx(m.paramID)
	}
	matchCluster.size++
}

func (m *Matcher) addSeqToPrefixTree(cluster *cluster) {
	tokenCount := len(cluster.tokenIDs)
	if tokenCount >= len(m.rootByLen) {
		m.rootByLen = append(m.rootByLen, make([]*node, tokenCount-len(m.rootByLen)+1)...)
	}
	if m.rootByLen[tokenCount] == nil {
		m.rootByLen[tokenCount] = newNode()
	}

	curNode := m.rootByLen[tokenCount]
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

		var next *node
		if n := curNode.children[tokenID]; n != nil {
			// Exact child exists.
			next = n
		} else if m.cfg.ParametrizeNumericTokens && hasNumbers(cluster.tokenStr[i]) {
			// Numeric token: always route to wildcard.
			next = curNode.children[m.paramID]
			if next == nil {
				next = newNode()
				curNode.children[m.paramID] = next
			}
		} else {
			// Non-numeric token not yet in tree.
			// Specific tokens get up to MaxChildren-1 slots; one slot is
			// reserved for the wildcard/param catch-all.
			wild := curNode.children[m.paramID]
			specificCount := len(curNode.children)
			if wild != nil {
				specificCount-- // wildcard already occupies its reserved slot
			}
			if specificCount < m.cfg.MaxChildren-1 {
				next = newNode()
				curNode.children[tokenID] = next
			} else {
				if wild == nil {
					wild = newNode()
					curNode.children[m.paramID] = wild
				}
				next = wild
			}
		}

		curNode = next
		curDepth++
	}
}
