package drain3

import (
	"cmp"
	"math"
	"slices"

	"github.com/lemire/constmap"
)

// Match returns template id, extracted args, and whether a match was found.
// Matcher-level Match* methods share one internal Session and follow the
// one-goroutine-per-Matcher rule; use NewSession for concurrent matching.
func (m *Matcher) Match(line string) (templateID int, args []string, ok bool) {
	return m.MatchInto(line, nil)
}

// MatchID returns just the template id and whether a match was found, without extracting args.
func (m *Matcher) MatchID(line string) (templateID int, ok bool) {
	s := m.session()
	if s == nil {
		return 0, false
	}
	return s.MatchID(line)
}

// MatchInto returns template id, extracted args into dst, and whether a match was found.
func (m *Matcher) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	s := m.session()
	if s == nil {
		return 0, nil, false
	}
	return s.MatchInto(line, dst)
}

// MatchExactInto returns a match only when every non-param template token
// exactly matches the input token at the same position; param positions act
// as wildcards. When several templates qualify, the most parametrized one is
// returned, ties broken by lowest template ID, so the result is deterministic.
func (m *Matcher) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool) {
	s := m.session()
	if s == nil {
		return 0, nil, false
	}
	return s.MatchExactInto(line, dst)
}

// session returns the Matcher-level default Session, rebuilding it if
// the Matcher was copied by value after freezing (rebuildFromTemplates
// publishes via *m = *next), which leaves the back-pointer stale.
func (m *Matcher) session() *Session {
	if m == nil || m.dictFrozen == nil {
		return nil
	}
	if m.defaultSession == nil || m.defaultSession.m != m {
		m.defaultSession = m.NewSession()
	}
	return m.defaultSession
}

func (s *Session) findMatch(line string) (cluster *cluster, tokens []string) {
	tokens, tokenCount, firstID, ok := s.tokenizeMatchLine(line)
	if !ok {
		return nil, nil
	}
	m := s.m
	// Fast path: prefilter narrows by token count + first/last token, then
	// scores candidates by string comparison, or — when the candidate set
	// is large enough to amortize resolving all tokens to dictionary IDs —
	// by uint64 comparison (see useIDScorerFor).
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := s.candidates
		if candidateIDs, owned, ok := m.prefilterCandidatesCompact(tokens, tokenCount, firstID, buf[:0]); ok {
			if owned && cap(candidateIDs) > cap(buf) {
				s.candidates = candidateIDs[:0:cap(candidateIDs)]
			}
			if m.useIDScorerFor(len(candidateIDs), tokenCount) {
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

func (s *Session) findExactMatch(line string) (cluster *cluster, tokens []string) {
	tokens, tokenCount, _, ok := s.tokenizeMatchLine(line)
	if !ok {
		return nil, nil
	}
	m := s.m
	if m.cfg.EnableMatchPrefilter && tokenCount < len(m.prefilterBuckets) {
		buf := s.candidates
		if candidateIDs, owned, ok := s.prefilterExactCandidatesCompact(tokens, tokenCount, buf[:0]); ok {
			if owned && cap(candidateIDs) > cap(buf) {
				s.candidates = candidateIDs[:0:cap(candidateIDs)]
			}
			if m.useIDScorerFor(len(candidateIDs), tokenCount) {
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
func (s *Session) tokenizeMatchLine(line string) (tokens []string, tokenCount int, firstID uint64, ok bool) {
	m := s.m
	if len(line) > m.cfg.MaxBytes {
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
		tokens, tokenCount = tokenizeWhitespaceCount(line, s.tok, m.cfg.MaxTokens)
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

// bbTarget returns the branch-and-bound score target for a candidate:
// once a qualifying best exists (bestScore >= needed), a candidate is
// only worth scoring while it can still beat it (tie on score wins only
// with more params). Exact mode keeps the plain threshold — its id
// tie-break can prefer an equal-score candidate.
func bbTarget(exact bool, needed, bestScore, bestParams, paramCount int) int {
	if exact || bestScore < needed {
		return needed
	}
	if paramCount > bestParams {
		return bestScore
	}
	return bestScore + 1
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
		target := bbTarget(exact, needed, maxScore, maxParamCount, paramCount)
		npIdx := cluster.nonParamIdx
		remaining := len(npIdx)
		for _, idx := range npIdx {
			if cIDs[idx] == tokenIDs[idx] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < target {
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
		target := bbTarget(exact, needed, maxScore, maxParamCount, paramCount)
		npIdx := c.nonParamIdx
		remaining := len(npIdx)
		anchor0 := c.anchor0
		anchor1 := c.anchor1
		if anchor0 >= 0 {
			if cStr[anchor0] == tokens[anchor0] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < target {
				continue
			}
		}
		if anchor1 >= 0 {
			if cStr[anchor1] == tokens[anchor1] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < target {
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
			if simTokens+remaining < target {
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
		target := bbTarget(exact, needed, maxScore, maxParamCount, paramCount)
		npIdx := c.nonParamIdx
		remaining := len(npIdx)
		anchor0 := c.anchor0
		anchor1 := c.anchor1
		if anchor0 >= 0 {
			if cIDs[anchor0] == ids[anchor0] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < target {
				continue
			}
		}
		if anchor1 >= 0 {
			if cIDs[anchor1] == ids[anchor1] {
				simTokens++
			}
			remaining--
			if simTokens+remaining < target {
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
			if simTokens+remaining < target {
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

// useIDScorerFor picks the candidate-verification strategy for one line.
// The ID scorer pays tokenCount hash lookups up front to make each
// candidate check a uint64 compare; the string scorer skips the lookups
// but pays a string compare per candidate check. Measured break-even is
// where the candidate count reaches the token count: below it the string
// scorer wins (real-world corpora: few candidates, long lines), above it
// the ID scorer wins by 3x+ (dense buckets: hundreds of candidates).
func (m *Matcher) useIDScorerFor(candidates, tokenCount int) bool {
	switch m.cfg.MatchScorer {
	case MatchScorerString:
		return false
	case MatchScorerID:
		return true
	default: // MatchScorerAuto
		return candidates >= tokenCount
	}
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
			combined := packAnchorIDs(firstID, lastID)
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
	maxProbe := 0
	for tc := range buckets {
		buckets[tc].buildAnchorIndex(anchorByTC[tc], anchorPairByTC[tc])
		if n := len(buckets[tc].probePos); n > maxProbe {
			maxProbe = n
		}
	}
	m.maxProbe = maxProbe

	m.prefilterBuckets = buckets
}

// buildAnchorIndex flattens the per-bucket anchor accumulator maps into
// the probe-slot representation described on prefilterBucket: distinct
// anchor positions get probePos slots, and each position (or position
// pair) owns a sorted, contiguous ID range for binary search.
func (b *prefilterBucket) buildAnchorIndex(singles map[anchorKey][]int, pairs map[anchorPairKey][]int) {
	slots := make(map[uint16]uint16, 8)
	slot := func(pos uint16) uint16 {
		if s, ok := slots[pos]; ok {
			return s
		}
		s := uint16(len(b.probePos))
		slots[pos] = s
		b.probePos = append(b.probePos, pos)
		return s
	}

	if len(singles) > 0 {
		keys := make([]anchorKey, 0, len(singles))
		for k := range singles {
			keys = append(keys, k)
		}
		slices.SortFunc(keys, compareAnchorKey)
		prevPos := ^uint16(0)
		for _, k := range keys {
			if k.pos != prevPos {
				b.anchorSlot = append(b.anchorSlot, slot(k.pos))
				b.anchorStart = append(b.anchorStart, int32(len(b.anchorIDs)))
				prevPos = k.pos
			}
			b.anchorIDs = append(b.anchorIDs, k.id)
			b.anchorVals = append(b.anchorVals, singles[k])
		}
		b.anchorStart = append(b.anchorStart, int32(len(b.anchorIDs)))
	}

	if len(pairs) > 0 {
		keys := make([]anchorPairKey, 0, len(pairs))
		for k := range pairs {
			keys = append(keys, k)
		}
		slices.SortFunc(keys, compareAnchorPairKey)
		prev := anchorPairKey{pos0: ^uint16(0), pos1: ^uint16(0)}
		for _, k := range keys {
			if k.pos0 != prev.pos0 || k.pos1 != prev.pos1 {
				b.anchorPairSlot = append(b.anchorPairSlot, [2]uint16{slot(k.pos0), slot(k.pos1)})
				b.anchorPairStart = append(b.anchorPairStart, int32(len(b.anchorPairIDs)))
			}
			packed := packAnchorIDs(k.id0, k.id1)
			groupStart := int(b.anchorPairStart[len(b.anchorPairStart)-1])
			if n := len(b.anchorPairIDs); n > groupStart && b.anchorPairIDs[n-1] == packed {
				// Packed collision (IDs beyond 32 bits): merge candidate
				// lists — extra candidates are rejected by the scorer.
				b.anchorPairVals[n-1] = append(b.anchorPairVals[n-1], pairs[k]...)
			} else {
				b.anchorPairIDs = append(b.anchorPairIDs, packed)
				b.anchorPairVals = append(b.anchorPairVals, pairs[k])
			}
			prev = k
		}
		b.anchorPairStart = append(b.anchorPairStart, int32(len(b.anchorPairIDs)))
	}
}

func packAnchorIDs(id0, id1 uint64) uint64 {
	return (id0 << 32) | (id1 & 0xFFFFFFFF)
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
			combined := packAnchorIDs(firstID, lastID)
			if group := searchSortedU64(b.flKeys, b.flVals, combined); len(group) > 0 {
				groups = append(groups, group)
			}
		}
	}
	return mergePrefilterGroups(groups, dst)
}

func (s *Session) prefilterExactCandidatesCompact(tokens []string, tokenCount int, dst []int) (candidates []int, owned bool, ok bool) {
	b := &s.m.prefilterBuckets[tokenCount]
	var groupBuf [16][]int
	groupCount := 1 + len(b.anchorSlot) + len(b.anchorPairSlot)
	groups := groupBuf[:0]
	if groupCount > len(groupBuf) {
		groups = make([][]int, 0, groupCount)
	}
	if len(b.exactAny) > 0 {
		groups = append(groups, b.exactAny)
	}

	// Resolve each distinct anchor position once; every probe below reads
	// its slot. Anchor positions are always < tokenCount: the bucket is
	// keyed by token count and anchors come from its own templates.
	dict := s.m.dictFrozen
	ids := s.probeIDs[:len(b.probePos)]
	for slotIdx, pos := range b.probePos {
		ids[slotIdx] = dict.Map(tokens[pos])
	}
	for i, slot := range b.anchorSlot {
		id := ids[slot]
		if id == constmap.NotFound {
			continue
		}
		lo, hi := b.anchorStart[i], b.anchorStart[i+1]
		if group := searchSortedU64(b.anchorIDs[lo:hi], b.anchorVals[lo:hi], id); len(group) > 0 {
			groups = append(groups, group)
		}
	}
	for i, sl := range b.anchorPairSlot {
		id0, id1 := ids[sl[0]], ids[sl[1]]
		if id0 == constmap.NotFound || id1 == constmap.NotFound {
			continue
		}
		lo, hi := b.anchorPairStart[i], b.anchorPairStart[i+1]
		if group := searchSortedU64(b.anchorPairIDs[lo:hi], b.anchorPairVals[lo:hi], packAnchorIDs(id0, id1)); len(group) > 0 {
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

func compareAnchorKey(a, b anchorKey) int {
	if c := cmp.Compare(a.pos, b.pos); c != 0 {
		return c
	}
	return cmp.Compare(a.id, b.id)
}

// compareAnchorPairKey orders by position pair, then by packed ID — the
// same key the probe path binary-searches — so each position group's ID
// range is search-ordered even when raw IDs exceed 32 bits.
func compareAnchorPairKey(a, b anchorPairKey) int {
	if c := cmp.Compare(a.pos0, b.pos0); c != 0 {
		return c
	}
	if c := cmp.Compare(a.pos1, b.pos1); c != 0 {
		return c
	}
	return cmp.Compare(packAnchorIDs(a.id0, a.id1), packAnchorIDs(b.id0, b.id1))
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
