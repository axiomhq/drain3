package drain3

// Session holds the mutable per-call state for matching. A trained
// Matcher is immutable; any number of goroutines may share it, one
// Session per goroutine. A Session must not be used concurrently.
type Session struct {
	m          *Matcher
	tok        []string
	candidates []int
	probeIDs   []uint64
}

// NewSession returns a Session for matching against m. It panics if m
// has not been trained (built via Train, TrainWithConfig, or
// NewMatcherFromTemplates).
func (m *Matcher) NewSession() *Session {
	if m == nil || m.dictFrozen == nil {
		panic("drain3: NewSession on an untrained Matcher")
	}
	return &Session{
		m:          m,
		tok:        make([]string, 0, m.cfg.MaxTokens),
		candidates: make([]int, 0, 1024),
		probeIDs:   make([]uint64, m.maxProbe),
	}
}

// Match returns template id, extracted args, and whether a match was found.
func (s *Session) Match(line string) (templateID int, args []string, ok bool) {
	return s.MatchInto(line, nil)
}

// MatchID returns just the template id and whether a match was found.
func (s *Session) MatchID(line string) (templateID int, ok bool) {
	cluster, _ := s.findMatch(line)
	if cluster == nil {
		return 0, false
	}
	return cluster.id, true
}

// MatchInto returns template id, extracted args into dst, and whether a
// match was found.
func (s *Session) MatchInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := s.findMatch(line)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, s.m.paramID, dst), true
}

// MatchExactInto returns a match only when every non-param template token
// exactly matches the input token at the same position; param positions
// act as wildcards. Ties break to the most parametrized template, then
// lowest template ID.
func (s *Session) MatchExactInto(line string, dst []string) (templateID int, args []string, ok bool) {
	cluster, tokens := s.findExactMatch(line)
	if cluster == nil {
		return 0, nil, false
	}
	return cluster.id, cluster.extractArgsInto(tokens, s.m.paramID, dst), true
}

// BatchResult holds MatchBatch output. Reuse it across calls: all
// slices are truncated and refilled, so a warm result allocates
// nothing. Args entries alias the input lines and keep them alive.
type BatchResult struct {
	IDs    []int32  // per line: matched template ID, 0 = miss
	ArgOff []int32  // prefix offsets, len(lines)+1: line i's args are Args[ArgOff[i]:ArgOff[i+1]]
	Args   []string // extracted args, back to back
}

func (r *BatchResult) reset(n int) {
	if cap(r.IDs) < n {
		r.IDs = make([]int32, 0, n)
		r.ArgOff = make([]int32, 0, n+1)
	}
	r.IDs = r.IDs[:0]
	r.ArgOff = append(r.ArgOff[:0], 0)
	r.Args = r.Args[:0]
}

// MatchBatch matches every line and writes struct-of-arrays results
// into dst, allocating a fresh BatchResult when dst is nil. Template
// IDs are dense and far below MaxInt32 in any trainable dictionary.
func (s *Session) MatchBatch(lines []string, dst *BatchResult) *BatchResult {
	if dst == nil {
		dst = &BatchResult{}
	}
	dst.reset(len(lines))
	paramID := s.m.paramID
	for _, line := range lines {
		cluster, tokens := s.findMatch(line)
		if cluster == nil {
			dst.IDs = append(dst.IDs, 0)
		} else {
			dst.IDs = append(dst.IDs, int32(cluster.id))
			dst.Args = cluster.appendArgs(dst.Args, tokens, paramID)
		}
		dst.ArgOff = append(dst.ArgOff, int32(len(dst.Args)))
	}
	return dst
}
