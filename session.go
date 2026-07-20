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
