package drain3

import (
	"testing"

	"github.com/bits-and-blooms/bitset"
)

func TestRenderPlanAppend(t *testing.T) {
	params := bitset.New(4)
	params.Set(1)
	params.Set(3)

	plan := NewRenderPlan(Template{
		Tokens:     []string{"service", "status"},
		Params:     params,
		TokenCount: 4,
	}, func(arg int) int {
		return []int{3, 4}[arg]
	})

	got := string(plan.Append(nil, func(arg int) string {
		return []string{"api", "200"}[arg]
	}))
	if got != "service api status 200" {
		t.Fatalf("rendered template: got %q", got)
	}
	if plan.MaxSize() != len("service api status 200")+1 {
		t.Fatalf("max size: got %d", plan.MaxSize())
	}
}

func TestRenderPlanAppendNoParams(t *testing.T) {
	plan := NewRenderPlan(Template{
		Tokens:     []string{"fixed", "message"},
		Params:     bitset.New(2),
		TokenCount: 2,
	}, nil)

	got := string(plan.Append(nil, nil))
	if got != "fixed message" {
		t.Fatalf("rendered template: got %q", got)
	}
	if plan.MaxSize() != len(got) {
		t.Fatalf("max size: got %d, want %d", plan.MaxSize(), len(got))
	}
}
