package drain3

import (
	"cmp"
	"math"
	"slices"

	"github.com/lemire/constmap"
)

// Match returns template id, extracted args, and whether a match was found.
func (m *Matcher) Match(line string) (templateID int, args []string, ok bool) {
	return m.MatchInto(line, nil)
}

// MatchID returns just the template id and whether a match was found, without extracting args.
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	cluster, _ := m.findMatch(line, m.scratchTok)
	if cluster == nil {
		return 0, false
	}
	return cluster.id, true
}

// MatchInto returns template id, extracted args into dst, and whether a match was found.
func (m *Matcher) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := m.findMatch(line, m.scratchTok)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, dst), true
}

func (m *Matcher) findMatch(line string, tokenBuf []string) (cluster *cluster, tokens []string) {
	if m == nil || len(line) > m.cfg.MaxBytes {
		return nil, nil
	}
	// Quick rejection: if no template has a param at position 0 and the
	// line's first token is unknown to the dictionary, no match is possible.
	// Only valid without ExtraDelimiters (where first space = first token boundary).
	if !m.hasParamFirst && len(m.cfg.ExtraDelimiters) == 0 {
		firstEnd := 0
		for firstEnd < len(line) && line[firstEnd] != ' ' {
			firstEnd++
		}
		if m.dictFrozen.Map(line[:firstEnd]) == constmap.NotFound {
			return nil, nil
		}
	}
	var tokenCount int
	if len(m.cfg.ExtraDelimiters) == 0 {
		tokens, tokenCount = tokenizeWhitespaceCount(line, tokenBuf, m.cfg.MaxTokens)
		if tokenCount > m.cfg.MaxTokens {
			return nil, nil
		}
		if tokenCount >= len(m.rootByLen) || m.rootByLen[tokenCount] == nil {
			return nil, nil
		}
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
		tokenCount = len(tokens)
		if tokenCount > m.cfg.MaxTokens {
			return nil, nil
		}
	}
	// Fast path: prefilter uses only first/last token IDs (2 lookups),
	// then scores candidates via direct string comparison — avoids
	// hashing all N tokens through the dictionary.
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := m.scratchCandidates
		if candidateIDs, ok := m.prefilterCandidatesCompact(tokens, tokenCount, buf[:0]); ok {
			if cap(candidateIDs) > cap(buf) {
				m.scratchCandidates = candidateIDs[:0:cap(candidateIDs)]
			}
			return m.fastMatchStrings(candidateIDs, tokens, m.cfg.MatchThreshold, true), tokens
		}
		return nil, tokens
	}
	// Slow path: tree search needs token IDs for navigation.
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

func (m *Matcher) treeSearchStrings(tokens []string, simTh float64, includeParams bool) *cluster {
	tokenCount := len(tokens)
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

		var nextNode *node
		if tokenID, ok := m.dictIDs[tokens[i]]; ok {
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

	return m.fastMatchStrings(curNode.clusterIDs, tokens, simTh, includeParams)
}

func (m *Matcher) fastMatch(clusterIDs []int, tokenIDs []uint64, simTh float64, includeParams bool) *cluster {
	nTokens := len(tokenIDs)
	needed := m.requiredScore(nTokens, simTh)
	maxScore := -1
	maxParamCount := -1
	var maxCluster *cluster
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
		npIdx := cluster.nonParamIdx
		remaining := len(npIdx)
		for _, idx := range npIdx {
			if cIDs[idx] == tokenIDs[idx] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < needed {
				break
			}
		}

		if simTokens > maxScore || (simTokens == maxScore && paramCount > maxParamCount) {
			maxScore = simTokens
			maxParamCount = paramCount
			maxCluster = cluster
		}
	}

	if maxScore >= needed {
		return maxCluster
	}
	return nil
}

// fastMatchStrings is like fastMatch but compares template token strings
// directly against input tokens, avoiding the cost of resolving all tokens
// to dictionary IDs. Used by the prefilter path.
//
// Anchor checks on the first and last non-param positions reject most
// candidates before entering the inner loop.
func (m *Matcher) fastMatchStrings(clusterIDs []int, tokens []string, simTh float64, includeParams bool) *cluster {
	nTokens := len(tokens)
	needed := m.requiredScore(nTokens, simTh)
	clusters := m.clusters

	// At threshold=1.0 every non-param token must match.
	// Anchor checks + bail on first mismatch, return first perfect match.
	if includeParams && simTh >= 1.0 {
	nextCandidate:
		for _, clusterID := range clusterIDs {
			c := clusters[clusterID]
			cStr := c.tokenStr
			if len(cStr) != nTokens {
				continue
			}
			// Anchor pre-rejection: cheap checks before full scan.
			if a := c.anchor0; a >= 0 && cStr[a] != tokens[a] {
				continue
			}
			if a := c.anchor1; a >= 0 && cStr[a] != tokens[a] {
				continue
			}
			for _, idx := range c.nonParamIdx {
				if cStr[idx] != tokens[idx] {
					continue nextCandidate
				}
			}
			return c
		}
		return nil
	}

	maxScore := -1
	maxParamCount := -1
	var maxCluster *cluster
	for _, clusterID := range clusterIDs {
		c := clusters[clusterID]
		cStr := c.tokenStr
		if len(cStr) != nTokens {
			continue
		}

		paramCount := c.paramCount
		simTokens := 0
		if includeParams {
			simTokens = paramCount
		}
		npIdx := c.nonParamIdx
		remaining := len(npIdx)
		anchor0 := c.anchor0
		anchor1 := c.anchor1
		if anchor0 >= 0 {
			if cStr[anchor0] == tokens[anchor0] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < needed {
				continue
			}
		}
		if anchor1 >= 0 {
			if cStr[anchor1] == tokens[anchor1] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < needed {
				continue
			}
		}
		for _, idx := range npIdx {
			idx := int(idx)
			if idx == anchor0 || idx == anchor1 {
				continue
			}
			if cStr[idx] == tokens[idx] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < needed {
				break
			}
		}

		if simTokens > maxScore || (simTokens == maxScore && paramCount > maxParamCount) {
			maxScore = simTokens
			maxParamCount = paramCount
			maxCluster = c
		}
	}

	if maxScore >= needed {
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
	dict := m.dictFrozen
	for i := range tokens {
		id := dict.Map(tokens[i])
		if id == constmap.NotFound {
			id = 0
		}
		dst[i] = id
	}
	return dst
}

func (m *Matcher) requiredScore(tokenCount int, simTh float64) int {
	if simTh == m.cfg.MatchThreshold && tokenCount >= 0 && tokenCount < len(m.matchNeeded) {
		return m.matchNeeded[tokenCount]
	}
	return int(math.Ceil(simTh * float64(tokenCount)))
}

func (m *Matcher) rebuildMatchNeeded() {
	m.matchNeeded = make([]int, len(m.rootByLen))
	for tokenCount := range len(m.matchNeeded) {
		m.matchNeeded[tokenCount] = int(math.Ceil(m.cfg.MatchThreshold * float64(tokenCount)))
	}
}

func (m *Matcher) rebuildMatchPrefilter() {
	if !m.cfg.EnableMatchPrefilter {
		m.prefilterBuckets = nil
		return
	}

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

	buckets := make([]prefilterBucket, maxLen+1)
	for tc, ids := range anyByTC {
		if tc < len(buckets) {
			buckets[tc].any = ids
		}
	}
	for tc, mm := range firstByTC {
		if tc < len(buckets) {
			buckets[tc].firstKeys, buckets[tc].firstVals = sortedU64Keys(mm)
		}
	}
	for tc, mm := range lastByTC {
		if tc < len(buckets) {
			buckets[tc].lastKeys, buckets[tc].lastVals = sortedU64Keys(mm)
		}
	}
	for tc, mm := range flByTC {
		if tc < len(buckets) {
			buckets[tc].flKeys, buckets[tc].flVals = sortedU64Keys(mm)
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
		dict := m.dictFrozen
		firstID := dict.Map(tokens[0])
		firstKnown := firstID != constmap.NotFound
		lastID := dict.Map(tokens[tokenCount-1])
		lastKnown := lastID != constmap.NotFound
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
