package drain3

import (
	"math"
	"strings"
)

// Match returns template id, extracted args, and whether a match was found.
func (m *Matcher) Match(line string) (templateID int, args []string, ok bool) {
	return m.MatchInto(line, nil)
}

// MatchInto returns template id, extracted args into dst, and whether a match was found.
func (m *Matcher) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	var tokenBuf [128]string
	cluster, tokens := m.findMatch(line, tokenBuf[:0])
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, m.paramID, dst), true
}

// MatchID returns template id and whether a match was found.
// Unlike Match, it does not extract arguments and performs zero allocations.
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	var tokenBuf [128]string
	cl, _ := m.findMatch(line, tokenBuf[:0])
	if cl == nil {
		return 0, false
	}
	return cl.id, true
}

func (m *Matcher) findMatch(line string, tokenBuf []string) (cluster *cluster, tokens []string) {
	if m == nil || len(line) > m.cfg.MaxBytes {
		return nil, nil
	}
	if len(m.cfg.ExtraDelimiters) == 0 {
		tokenCount := 1 + strings.Count(line, " ")
		if tokenCount > m.cfg.MaxTokens {
			return nil, nil
		}
		if tokenCount >= len(m.rootByLen) || m.rootByLen[tokenCount] == nil {
			return nil, nil
		}
		tokens = tokenizeWhitespaceInto(line, tokenBuf)
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
	}
	// Resolve tokens to dictionary IDs. Unknown tokens get 0 (guaranteed
	// mismatch during both tree navigation and scoring).
	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.treeSearch(tokenIDs, m.cfg.MatchThreshold, true), tokens
}

func (m *Matcher) treeSearch(tokenIDs []uint64, simTh float64, includeParams bool) *cluster {
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
		if curDepth >= maxDepth || curDepth == tokenCount {
			break
		}

		nextNode := curNode.children[tokenIDs[i]]
		if nextNode == nil {
			nextNode = curNode.children[m.paramID]
		}
		if nextNode == nil {
			return nil
		}

		curNode = nextNode
		curDepth++
	}

	return m.fastMatch(curNode.clusterIDs, tokenIDs, simTh, includeParams)
}

func (m *Matcher) fastMatch(clusterIDs []int, tokenIDs []uint64, simTh float64, includeParams bool) *cluster {
	nTokens := len(tokenIDs)
	requiredScore := requiredScore(nTokens, simTh)
	maxScore := -1
	maxParamCount := -1
	var maxCluster *cluster
	paramID := m.paramID
	clusters := m.clusters

	for _, clusterID := range clusterIDs {
		cluster := clusters[clusterID]
		cIDs := cluster.tokenIDs

		// Quick length check.
		if len(cIDs) != nTokens {
			continue
		}

		// Score: count matching non-param tokens (plus all params if includeParams).
		paramCount := cluster.paramCount
		simTokens := 0
		if includeParams {
			simTokens = paramCount
		}
		remaining := nTokens - paramCount
		for i, id := range cIDs {
			if id == paramID {
				continue
			}
			if id == tokenIDs[i] {
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
		dst[i] = m.dictIDs[tokens[i]] // unknown tokens get 0, guaranteed mismatch
	}
	return dst
}

func requiredScore(tokenCount int, simTh float64) int {
	return int(math.Ceil(simTh * float64(tokenCount)))
}

