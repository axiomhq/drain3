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
	return cluster.id, cluster.extractArgsInto(tokens, m.paramID, dst), true
}

// MatchExactInto returns a match only when every non-param template token
// exactly matches the input token at the same position; param positions act
// as wildcards. When several templates qualify, the most parametrized one is
// returned, ties broken by lowest template ID, so the result is deterministic.
func (m *Matcher) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := m.findExactMatch(line, m.scratchTok)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, m.paramID, dst), true
}

func (m *Matcher) findMatch(line string, tokenBuf []string) (cluster *cluster, tokens []string) {
	tokens, tokenCount, firstID, ok := m.tokenizeMatchLine(line, tokenBuf)
	if !ok {
		return nil, nil
	}
	// Fast path: prefilter narrows by token count + first/last token, then
	// resolves all tokens to dictionary IDs once and scores by uint64
	// comparison instead of per-token string memequal.
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := m.scratchCandidates
		if candidateIDs, owned, ok := m.prefilterCandidatesCompact(tokens, tokenCount, firstID, buf[:0]); ok {
			if owned && cap(candidateIDs) > cap(buf) {
				m.scratchCandidates = candidateIDs[:0:cap(candidateIDs)]
			}
			if m.useIDScorer {
				var idBuf [128]uint64
				ids := m.lookupTokenIDsPartial(tokens, idBuf[:0])
				return m.fastMatchIDs(candidateIDs, ids, m.cfg.MatchThreshold, true, false), tokens
			}
			return m.fastMatchStrings(candidateIDs, tokens, m.cfg.MatchThreshold, true, false), tokens
		}
		return nil, tokens
	}
	// Slow path: tree search needs token IDs for navigation.
	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.treeSearch(tokenIDs, m.cfg.MatchThreshold, true, false), tokens
}

func (m *Matcher) findExactMatch(line string, tokenBuf []string) (cluster *cluster, tokens []string) {
	tokens, tokenCount, _, ok := m.tokenizeMatchLine(line, tokenBuf)
	if !ok {
		return nil, nil
	}
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := m.scratchCandidates
		if candidateIDs, owned, ok := m.prefilterExactCandidatesCompact(tokens, tokenCount, buf[:0]); ok {
			if owned && cap(candidateIDs) > cap(buf) {
				m.scratchCandidates = candidateIDs[:0:cap(candidateIDs)]
			}
			if m.useIDScorer {
				var idBuf [128]uint64
				ids := m.lookupTokenIDsPartial(tokens, idBuf[:0])
				return m.fastMatchIDs(candidateIDs, ids, 1.0, true, true), tokens
			}
			return m.fastMatchStrings(candidateIDs, tokens, 1.0, true, true), tokens
		}
		return nil, tokens
	}

	var tokenIDBuf [128]uint64
	tokenIDs := m.lookupTokenIDsPartial(tokens, tokenIDBuf[:0])
	return m.treeSearch(tokenIDs, 1.0, true, true), tokens
}

// tokenizeMatchLine tokenizes a line for matching. firstID returns the
// dictionary ID of the first token when the quick-reject path resolved it
// (so the prefilter can reuse it instead of hashing the same token again);
// otherwise it is constmap.NotFound, meaning "caller must resolve".
func (m *Matcher) tokenizeMatchLine(line string, tokenBuf []string) (tokens []string, tokenCount int, firstID uint64, ok bool) {
	if m == nil || len(line) > m.cfg.MaxBytes {
		return nil, 0, constmap.NotFound, false
	}
	firstID = constmap.NotFound
	// Quick rejection: if no template has a param at position 0 and the
	// line's first token is unknown to the dictionary, no match is possible.
	// Only valid without ExtraDelimiters (where first space = first token
	// boundary). The resolved ID is threaded out for the prefilter to reuse.
	if !m.hasParamFirst && len(m.cfg.ExtraDelimiters) == 0 {
		firstEnd := 0
		for firstEnd < len(line) && line[firstEnd] != ' ' {
			firstEnd++
		}
		id := m.dictFrozen.Map(line[:firstEnd])
		if id == constmap.NotFound {
			return nil, 0, constmap.NotFound, false
		}
		firstID = id
	}
	if len(m.cfg.ExtraDelimiters) == 0 {
		tokens, tokenCount = tokenizeWhitespaceCount(line, tokenBuf, m.cfg.MaxTokens)
		if tokenCount > m.cfg.MaxTokens {
			return nil, 0, constmap.NotFound, false
		}
		if tokenCount >= len(m.rootByLen) || m.rootByLen[tokenCount] == nil {
			return nil, 0, constmap.NotFound, false
		}
	} else {
		tokens = tokenize(line, m.cfg.ExtraDelimiters)
		tokenCount = len(tokens)
		if tokenCount > m.cfg.MaxTokens {
			return nil, 0, constmap.NotFound, false
		}
	}
	return tokens, tokenCount, firstID, true
}

func (m *Matcher) treeSearch(tokenIDs []uint64, simTh float64, includeParams, exact bool) *cluster {
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

	return m.fastMatch(curNode.clusterIDs, tokenIDs, simTh, includeParams, exact)
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

	return m.fastMatchStrings(curNode.clusterIDs, tokens, simTh, includeParams, false)
}

func (m *Matcher) fastMatch(clusterIDs []int, tokenIDs []uint64, simTh float64, includeParams, exact bool) *cluster {
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

		better := simTokens > maxScore || (simTokens == maxScore && paramCount > maxParamCount)
		// Exact (MatchExactInto) path only: break score/paramCount ties by
		// lowest cluster id so the result is independent of iteration order
		// and matches the prefilter path. Normal Match is left unchanged.
		if exact && !better && maxCluster != nil &&
			simTokens == maxScore && paramCount == maxParamCount &&
			cluster.id < maxCluster.id {
			better = true
		}
		if better {
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
func (m *Matcher) fastMatchStrings(clusterIDs []int, tokens []string, simTh float64, includeParams, exact bool) *cluster {
	nTokens := len(tokens)
	needed := m.requiredScore(nTokens, simTh)
	clusters := m.clusters

	// Exact (MatchExactInto) path: every non-param token must match. Among
	// all fully-matching candidates pick the most parametrized template,
	// ties broken by lowest cluster id, so the result is deterministic
	// regardless of candidate iteration order.
	if includeParams && exact {
		var best *cluster
	nextExact:
		for _, clusterID := range clusterIDs {
			c := clusters[clusterID]
			cStr := c.tokenStr
			if len(cStr) != nTokens {
				continue
			}
			if a := c.anchor0; a >= 0 && cStr[a] != tokens[a] {
				continue
			}
			if a := c.anchor1; a >= 0 && cStr[a] != tokens[a] {
				continue
			}
			for _, idx := range c.nonParamIdx {
				if cStr[idx] != tokens[idx] {
					continue nextExact
				}
			}
			if best == nil || c.paramCount > best.paramCount ||
				(c.paramCount == best.paramCount && c.id < best.id) {
				best = c
			}
		}
		return best
	}

	// Normal Match at threshold 1.0: every non-param token must match.
	// Return the first fully-matching candidate (fast short-circuit).
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

// fastMatchIDs mirrors fastMatchStrings but compares interned token IDs
// (uint64) instead of token strings, eliminating per-token memequal. The
// caller resolves line tokens to IDs once; unknown tokens are 0, which
// never equals a valid literal ID, so results are identical to the string
// comparison (the dictionary is injective).
func (m *Matcher) fastMatchIDs(clusterIDs []int, ids []uint64, simTh float64, includeParams, exact bool) *cluster {
	nTokens := len(ids)
	needed := m.requiredScore(nTokens, simTh)
	clusters := m.clusters

	// Exact (MatchExactInto) path: every non-param token must match. Pick
	// the most parametrized full match, ties broken by lowest cluster id.
	if includeParams && exact {
		var best *cluster
	nextExact:
		for _, clusterID := range clusterIDs {
			c := clusters[clusterID]
			cIDs := c.tokenIDs
			if len(cIDs) != nTokens {
				continue
			}
			if a := c.anchor0; a >= 0 && cIDs[a] != ids[a] {
				continue
			}
			if a := c.anchor1; a >= 0 && cIDs[a] != ids[a] {
				continue
			}
			for _, idx := range c.nonParamIdx {
				if cIDs[idx] != ids[idx] {
					continue nextExact
				}
			}
			if best == nil || c.paramCount > best.paramCount ||
				(c.paramCount == best.paramCount && c.id < best.id) {
				best = c
			}
		}
		return best
	}

	// Normal Match at threshold 1.0: return the first full match.
	if includeParams && simTh >= 1.0 {
	nextCandidate:
		for _, clusterID := range clusterIDs {
			c := clusters[clusterID]
			cIDs := c.tokenIDs
			if len(cIDs) != nTokens {
				continue
			}
			if a := c.anchor0; a >= 0 && cIDs[a] != ids[a] {
				continue
			}
			if a := c.anchor1; a >= 0 && cIDs[a] != ids[a] {
				continue
			}
			for _, idx := range c.nonParamIdx {
				if cIDs[idx] != ids[idx] {
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
		cIDs := c.tokenIDs
		if len(cIDs) != nTokens {
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
			if cIDs[anchor0] == ids[anchor0] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < needed {
				continue
			}
		}
		if anchor1 >= 0 {
			if cIDs[anchor1] == ids[anchor1] {
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
			if cIDs[idx] == ids[idx] {
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
		anyByTC        = make(map[int][]int)
		exactAnyByTC   = make(map[int][]int)
		firstByTC      = make(map[int]map[uint64][]int)
		lastByTC       = make(map[int]map[uint64][]int)
		flByTC         = make(map[int]map[uint64][]int)
		anchorByTC     = make(map[int]map[anchorKey][]int)
		anchorPairByTC = make(map[int]map[anchorPairKey][]int)
		maxLen         = 0
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
			exactAnyByTC[0] = append(exactAnyByTC[0], id)
			continue
		}

		firstID := cluster.tokenIDs[0]
		lastID := cluster.tokenIDs[tokenCount-1]
		firstIsParam := firstID == m.paramID
		lastIsParam := lastID == m.paramID
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

		switch {
		case cluster.anchor0 < 0:
			exactAnyByTC[tokenCount] = append(exactAnyByTC[tokenCount], id)
		case cluster.anchor1 < 0:
			if cluster.anchor0 > math.MaxUint16 {
				// Position not representable in the uint16 anchor key;
				// fall back to the always-candidate group.
				exactAnyByTC[tokenCount] = append(exactAnyByTC[tokenCount], id)
				break
			}
			if anchorByTC[tokenCount] == nil {
				anchorByTC[tokenCount] = make(map[anchorKey][]int)
			}
			key := anchorKey{
				pos: uint16(cluster.anchor0),
				id:  cluster.tokenIDs[cluster.anchor0],
			}
			anchorByTC[tokenCount][key] = append(anchorByTC[tokenCount][key], id)
		default:
			if cluster.anchor0 > math.MaxUint16 || cluster.anchor1 > math.MaxUint16 {
				exactAnyByTC[tokenCount] = append(exactAnyByTC[tokenCount], id)
				break
			}
			if anchorPairByTC[tokenCount] == nil {
				anchorPairByTC[tokenCount] = make(map[anchorPairKey][]int)
			}
			key := anchorPairKey{
				pos0: uint16(cluster.anchor0),
				pos1: uint16(cluster.anchor1),
				id0:  cluster.tokenIDs[cluster.anchor0],
				id1:  cluster.tokenIDs[cluster.anchor1],
			}
			anchorPairByTC[tokenCount][key] = append(anchorPairByTC[tokenCount][key], id)
		}
	}

	buckets := make([]prefilterBucket, maxLen+1)
	for tc, ids := range anyByTC {
		if tc < len(buckets) {
			buckets[tc].any = ids
		}
	}
	for tc, ids := range exactAnyByTC {
		if tc < len(buckets) {
			buckets[tc].exactAny = ids
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
	for tc, mm := range anchorByTC {
		if tc < len(buckets) {
			buckets[tc].anchorKeys, buckets[tc].anchorVals = sortedAnchorKeys(mm)
			buckets[tc].anchorPos = uniqueAnchorPositions(buckets[tc].anchorKeys)
		}
	}
	for tc, mm := range anchorPairByTC {
		if tc < len(buckets) {
			buckets[tc].anchorPairKeys, buckets[tc].anchorPairVals = sortedAnchorPairKeys(mm)
			buckets[tc].anchorPairPos = uniqueAnchorPairPositions(buckets[tc].anchorPairKeys)
		}
	}

	m.prefilterBuckets = buckets
}

func (m *Matcher) prefilterCandidatesCompact(tokens []string, tokenCount int, firstID uint64, dst []int) (candidates []int, owned bool, ok bool) {
	b := &m.prefilterBuckets[tokenCount]
	var groupBuf [4][]int
	groups := groupBuf[:0]
	if len(b.any) > 0 {
		groups = append(groups, b.any)
	}

	if tokenCount > 0 {
		dict := m.dictFrozen
		// firstID is reused from the quick-reject when it resolved it;
		// otherwise it is the NotFound sentinel and must be looked up here.
		if firstID == constmap.NotFound {
			firstID = dict.Map(tokens[0])
		}
		firstKnown := firstID != constmap.NotFound
		lastID := dict.Map(tokens[tokenCount-1])
		lastKnown := lastID != constmap.NotFound
		if firstKnown {
			if group := searchSortedU64(b.firstKeys, b.firstVals, firstID); len(group) > 0 {
				groups = append(groups, group)
			}
		}
		if lastKnown {
			if group := searchSortedU64(b.lastKeys, b.lastVals, lastID); len(group) > 0 {
				groups = append(groups, group)
			}
		}
		if firstKnown && lastKnown {
			combined := (firstID << 32) | (lastID & 0xFFFFFFFF)
			if group := searchSortedU64(b.flKeys, b.flVals, combined); len(group) > 0 {
				groups = append(groups, group)
			}
		}
	}
	return mergePrefilterGroups(groups, dst)
}

func (m *Matcher) prefilterExactCandidatesCompact(tokens []string, tokenCount int, dst []int) (candidates []int, owned bool, ok bool) {
	b := &m.prefilterBuckets[tokenCount]
	var groupBuf [16][]int
	groupCount := 1 + len(b.anchorPos) + len(b.anchorPairPos)
	groups := groupBuf[:0]
	if groupCount > len(groupBuf) {
		groups = make([][]int, 0, groupCount)
	}
	if len(b.exactAny) > 0 {
		groups = append(groups, b.exactAny)
	}

	dict := m.dictFrozen
	for _, pos := range b.anchorPos {
		if int(pos) >= tokenCount {
			continue
		}
		id := dict.Map(tokens[pos])
		if id == constmap.NotFound {
			continue
		}
		if group := searchSortedAnchor(b.anchorKeys, b.anchorVals, anchorKey{pos: pos, id: id}); len(group) > 0 {
			groups = append(groups, group)
		}
	}
	for _, pos := range b.anchorPairPos {
		if int(pos.pos0) >= tokenCount || int(pos.pos1) >= tokenCount {
			continue
		}
		id0 := dict.Map(tokens[pos.pos0])
		if id0 == constmap.NotFound {
			continue
		}
		id1 := dict.Map(tokens[pos.pos1])
		if id1 == constmap.NotFound {
			continue
		}
		key := anchorPairKey{
			pos0: pos.pos0,
			pos1: pos.pos1,
			id0:  id0,
			id1:  id1,
		}
		if group := searchSortedAnchorPair(b.anchorPairKeys, b.anchorPairVals, key); len(group) > 0 {
			groups = append(groups, group)
		}
	}
	return mergePrefilterGroups(groups, dst)
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

func sortedAnchorKeys(m map[anchorKey][]int) ([]anchorKey, [][]int) {
	keys := make([]anchorKey, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, compareAnchorKey)
	vals := make([][]int, len(keys))
	for i, k := range keys {
		vals[i] = m[k]
	}
	return keys, vals
}

func sortedAnchorPairKeys(m map[anchorPairKey][]int) ([]anchorPairKey, [][]int) {
	keys := make([]anchorPairKey, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, compareAnchorPairKey)
	vals := make([][]int, len(keys))
	for i, k := range keys {
		vals[i] = m[k]
	}
	return keys, vals
}

func uniqueAnchorPositions(keys []anchorKey) []uint16 {
	if len(keys) == 0 {
		return nil
	}
	positions := make([]uint16, 0, len(keys))
	last := uint16(^uint16(0))
	for _, key := range keys {
		if key.pos == last {
			continue
		}
		positions = append(positions, key.pos)
		last = key.pos
	}
	return positions
}

func uniqueAnchorPairPositions(keys []anchorPairKey) []anchorPairPos {
	if len(keys) == 0 {
		return nil
	}
	positions := make([]anchorPairPos, 0, len(keys))
	last := anchorPairPos{pos0: ^uint16(0), pos1: ^uint16(0)}
	for _, key := range keys {
		pos := anchorPairPos{pos0: key.pos0, pos1: key.pos1}
		if pos == last {
			continue
		}
		positions = append(positions, pos)
		last = pos
	}
	return positions
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

func searchSortedAnchor(keys []anchorKey, vals [][]int, target anchorKey) []int {
	lo, hi := 0, len(keys)
	for lo < hi {
		mid := lo + (hi-lo)/2
		c := compareAnchorKey(keys[mid], target)
		if c == 0 {
			return vals[mid]
		}
		if c < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return nil
}

func searchSortedAnchorPair(keys []anchorPairKey, vals [][]int, target anchorPairKey) []int {
	lo, hi := 0, len(keys)
	for lo < hi {
		mid := lo + (hi-lo)/2
		c := compareAnchorPairKey(keys[mid], target)
		if c == 0 {
			return vals[mid]
		}
		if c < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return nil
}

func compareAnchorKey(a, b anchorKey) int {
	if c := cmp.Compare(a.pos, b.pos); c != 0 {
		return c
	}
	return cmp.Compare(a.id, b.id)
}

func compareAnchorPairKey(a, b anchorPairKey) int {
	if c := cmp.Compare(a.pos0, b.pos0); c != 0 {
		return c
	}
	if c := cmp.Compare(a.pos1, b.pos1); c != 0 {
		return c
	}
	if c := cmp.Compare(a.id0, b.id0); c != 0 {
		return c
	}
	return cmp.Compare(a.id1, b.id1)
}

// mergePrefilterGroups returns the union of the candidate groups. The owned
// return reports whether the returned slice is backed by dst (or a fresh
// allocation) and is therefore safe for the caller to retain as a scratch
// buffer. When a single non-empty group is returned it aliases prefilter
// index memory (owned == false) and must not be written to or retained.
func mergePrefilterGroups(groups [][]int, dst []int) (candidates []int, owned bool, ok bool) {
	var single []int
	total := 0
	nonEmpty := 0
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		nonEmpty++
		single = group
		total += len(group)
	}
	if nonEmpty == 0 {
		return nil, false, false
	}
	if nonEmpty == 1 {
		return single, false, true
	}

	out := dst[:0]
	if cap(out) < total {
		out = make([]int, 0, total)
	}
	for _, group := range groups {
		out = append(out, group...)
	}
	return out, true, true
}
