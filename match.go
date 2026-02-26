package drain3

import (
	"cmp"
	"math"
	"slices"
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
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	templateID, _, ok = m.Match(line)
	return
}

func (m *Matcher) findMatch(line string, tokenBuf []string) (cluster *cluser, tokens []string) {
	if m == nil || len(line) > m.cfg.MaxBytes {
		return nil, nil
	}
	if len(m.cfg.ExtraDelimiters) == 0 {
		tokenCount := 1 + strings.Count(line, " ")
		if tokenCount > m.cfg.MaxTokens {
			return nil, nil
		}
		if tokenCount >= len(m.flatRootByLen) || m.flatRootByLen[tokenCount] == 0 {
			return nil, nil
		}
		tokens = tokenizeWhitespaceInto(line, tokenBuf)
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
	}
	return m.treeSearchMatch(tokens, m.cfg.MatchThreshold), tokens
}

func (m *Matcher) treeSearchMatch(tokens []string, simTh float64) *cluser {
	tokenCount := len(tokens)

	// Fast length-based rejection: no templates exist with this token count.
	if tokenCount >= len(m.flatRootByLen) || m.flatRootByLen[tokenCount] == 0 {
		return nil
	}

	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		var candidateBuf [64]int
		if candidateIDs, ok := m.prefilterCandidatesCompact(tokens, tokenCount, candidateBuf[:0]); ok {
			return m.fastMatchAdaptive(candidateIDs, tokens, simTh)
		}
	}

	curNodeIdx := m.flatRootByLen[tokenCount]
	if curNodeIdx == 0 {
		return nil
	}
	curNode := &m.flatNodes[curNodeIdx]

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

		nextNodeIdx := uint32(0)
		if tokenID, ok := m.dictIDs[tokens[i]]; ok {
			nextNodeIdx = flatNodeChild(curNode, tokenID)
		}
		if nextNodeIdx == 0 {
			nextNodeIdx = flatNodeChild(curNode, m.paramID)
		}
		if nextNodeIdx == 0 {
			return nil
		}

		curNode = &m.flatNodes[nextNodeIdx]
		curDepth++
	}

	return m.fastMatchAdaptive(curNode.clusterIDs, tokens, simTh)
}

func (m *Matcher) fastMatch(clusterIDs []int, tokenIDs []uint64, simTh float64, includeParams bool) *cluser {
	nTokens := len(tokenIDs)
	requiredScore := m.requiredScore(nTokens, simTh)
	maxScore := -1
	maxParamCount := -1
	var maxCluster *cluser
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

func (m *Matcher) fastMatchAdaptive(clusterIDs []int, tokens []string, simTh float64) *cluser {
	if len(clusterIDs) == 0 {
		return nil
	}
	// Always resolve to IDs. Unknown tokens get sentinel value 0 and are
	// counted as automatic mismatches (they can't match any dictionary token).
	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.fastMatch(clusterIDs, tokenIDs, simTh, true)
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

func (m *Matcher) rebuildFlatTree() {
	m.flatRootByLen = make([]uint32, len(m.rootByLen))
	m.flatNodes = make([]FlatNode, 1) // index 0 means "missing node"
	if len(m.rootByLen) == 0 {
		return
	}

	seen := make(map[*Node]uint32, len(m.rootByLen))
	for tokenCount, root := range m.rootByLen {
		if root == nil {
			continue
		}
		m.flatRootByLen[tokenCount] = flattenNode(root, &m.flatNodes, seen)
	}
}

func flattenNode(node *Node, nodes *[]FlatNode, seen map[*Node]uint32) uint32 {
	if idx, ok := seen[node]; ok {
		return idx
	}

	idx := uint32(len(*nodes))
	seen[node] = idx
	*nodes = append(*nodes, FlatNode{
		clusterIDs: append([]int(nil), node.clusterIDs...),
	})

	if len(node.children) == 0 {
		return idx
	}

	keys := make([]uint64, 0, len(node.children))
	for key := range node.children {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b uint64) int { return cmp.Compare(a, b) })

	children := make([]uint32, len(keys))
	for i, key := range keys {
		children[i] = flattenNode(node.children[key], nodes, seen)
	}

	flat := &(*nodes)[idx]
	flat.childKeys = keys
	flat.childIndexes = children
	return idx
}

func flatNodeChild(node *FlatNode, tokenID uint64) uint32 {
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
		m.prefilterBuckets = nil
		return
	}

	// Local maps keyed by token count for grouping clusters.
	var (
		anyByTC   = make(map[int][]int)
		firstByTC = make(map[int]map[uint64][]int)
		lastByTC  = make(map[int]map[uint64][]int)
		flByTC    = make(map[int]map[uint64][]int)
		maxLen    = 0
	)

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
			anyByTC[0] = append(anyByTC[0], id)
			continue
		}

		var (
			firstID      = cluster.tokenIDs[0]
			lastID       = cluster.tokenIDs[tokenCount-1]
			firstIsParam = firstID == m.paramID
			lastIsParam  = lastID == m.paramID
		)
		switch {
		case firstIsParam && lastIsParam:
			anyByTC[tokenCount] = append(anyByTC[tokenCount], id)
		case !firstIsParam && lastIsParam:
			if firstByTC[tokenCount] == nil {
				firstByTC[tokenCount] = make(map[uint64][]int)
			}
			firstByTC[tokenCount][firstID] = append(firstByTC[tokenCount][firstID], id)
		case firstIsParam && !lastIsParam:
			if lastByTC[tokenCount] == nil {
				lastByTC[tokenCount] = make(map[uint64][]int)
			}
			lastByTC[tokenCount][lastID] = append(lastByTC[tokenCount][lastID], id)
		default:
			if flByTC[tokenCount] == nil {
				flByTC[tokenCount] = make(map[uint64][]int)
			}
			combined := (firstID << 32) | (lastID & 0xFFFFFFFF)
			flByTC[tokenCount][combined] = append(flByTC[tokenCount][combined], id)
		}
	}

	// Convert to sorted prefilterBucket arrays.
	buckets := make([]prefilterBucket, maxLen+1)
	for tc, ids := range anyByTC {
		if tc < len(buckets) {
			buckets[tc].any = ids
		}
	}
	for tc, m := range firstByTC {
		if tc < len(buckets) {
			buckets[tc].firstKeys, buckets[tc].firstVals = sortedU64Keys(m)
		}
	}
	for tc, m := range lastByTC {
		if tc < len(buckets) {
			buckets[tc].lastKeys, buckets[tc].lastVals = sortedU64Keys(m)
		}
	}
	for tc, m := range flByTC {
		if tc < len(buckets) {
			buckets[tc].flKeys, buckets[tc].flVals = sortedU64Keys(m)
		}
	}

	m.prefilterBuckets = buckets
}

func (m *Matcher) prefilterCandidatesCompact(tokens []string, tokenCount int, dst []int) ([]int, bool) {
	b := &m.prefilterBuckets[tokenCount]
	any := b.any

	var first []int
	var last []int
	var firstLast []int
	if tokenCount > 0 {
		firstID, firstKnown := m.dictIDs[tokens[0]]
		lastID, lastKnown := m.dictIDs[tokens[tokenCount-1]]
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

func sortedU64Keys(m map[uint64][]int) ([]uint64, [][]int) {
	keys := make([]uint64, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b uint64) int { return cmp.Compare(a, b) })
	vals := make([][]int, len(keys))
	for i, k := range keys {
		vals[i] = m[k]
	}
	return keys, vals
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
