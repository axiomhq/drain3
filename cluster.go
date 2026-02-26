package drain3

type cluser struct {
	id         int
	size       int
	paramCount int
	tokenIDs   []uint64
	tokenStr   []string
	// Pre-computed non-param indices for fast scoring.
	nonParamIdx []uint16
}

func newCluster(id int, tokenStr []string, tokenIDs []uint64, size int, paramID uint64) *cluser {
	c := &cluser{id: id, size: size, tokenStr: tokenStr, tokenIDs: tokenIDs}
	np := make([]uint16, 0, len(tokenIDs))
	for i, tid := range tokenIDs {
		if tid == paramID {
			c.paramCount++
		} else {
			np = append(np, uint16(i))
		}
	}
	c.nonParamIdx = np
	return c
}

func (c *cluser) extractArgsInto(lineTokens []string, paramID uint64, dst []string) []string {
	if len(c.tokenIDs) == 0 || len(lineTokens) == 0 || c.paramCount == 0 {
		return nil
	}
	limit := len(c.tokenIDs)
	if len(lineTokens) < limit {
		limit = len(lineTokens)
	}
	paramCount := c.paramCount
	if paramCount > limit {
		paramCount = limit
	}
	args := dst[:0]
	if cap(args) < paramCount {
		args = make([]string, 0, paramCount)
	}
	for i := 0; i < limit; i++ {
		if c.tokenIDs[i] == paramID {
			args = append(args, lineTokens[i])
		}
	}
	if len(args) == 0 {
		return nil
	}
	return args
}
