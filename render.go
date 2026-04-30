package drain3

// RenderPlan is a precomputed recipe for rendering a template with supplied
// parameter values.
type RenderPlan struct {
	head     []byte
	segments []renderSegment
	maxSize  int
}

type renderSegment struct {
	argIdx int
	tail   []byte
}

// NewRenderPlan builds a render plan for t.
//
// maxArgLen is optional. When provided, it is called once for each parameter
// position and included in MaxSize.
func NewRenderPlan(t Template, maxArgLen func(arg int) int) RenderPlan {
	var head, cur []byte
	var segments []renderSegment
	argIdx, tokIdx := 0, 0
	for i := range t.TokenCount {
		if i > 0 {
			cur = append(cur, ' ')
		}
		if t.Params.Test(uint(i)) {
			if len(segments) == 0 {
				head = cur
			} else {
				segments[len(segments)-1].tail = cur
			}
			segments = append(segments, renderSegment{argIdx: argIdx})
			cur = nil
			argIdx++
		} else {
			cur = append(cur, t.Tokens[tokIdx]...)
			tokIdx++
		}
	}
	if len(segments) == 0 {
		head = cur
	} else {
		segments[len(segments)-1].tail = cur
	}

	maxSize := len(head)
	for _, segment := range segments {
		maxSize += len(segment.tail)
		if maxArgLen != nil {
			maxSize += maxArgLen(segment.argIdx)
		}
	}
	return RenderPlan{head: head, segments: segments, maxSize: maxSize}
}

// MaxSize returns the upper bound calculated by NewRenderPlan.
func (p RenderPlan) MaxSize() int {
	return p.maxSize
}

// Append appends the rendered template to dst.
//
// arg is called with a zero-based parameter index. If arg is nil, parameters
// render as empty strings.
func (p RenderPlan) Append(dst []byte, arg func(arg int) string) []byte {
	dst = append(dst, p.head...)
	for _, segment := range p.segments {
		if arg != nil {
			dst = append(dst, arg(segment.argIdx)...)
		}
		dst = append(dst, segment.tail...)
	}
	return dst
}
