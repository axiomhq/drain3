package drain3

import (
	"testing"

	"github.com/bits-and-blooms/bitset"
)

func TestRenderPlanAppend(t *testing.T) {
	params := bitset.New(4)
	params.Set(1)
	params.Set(3)

	plan, err := NewRenderPlan(Template{
		Tokens:     []string{"service", "status"},
		Params:     params,
		TokenCount: 4,
	}, func(arg int) int {
		return []int{3, 4}[arg]
	})
	if err != nil {
		t.Fatal(err)
	}

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
	plan, err := NewRenderPlan(Template{
		Tokens:     []string{"fixed", "message"},
		Params:     bitset.New(2),
		TokenCount: 2,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	got := string(plan.Append(nil, nil))
	if got != "fixed message" {
		t.Fatalf("rendered template: got %q", got)
	}
	if plan.MaxSize() != len(got) {
		t.Fatalf("max size: got %d, want %d", plan.MaxSize(), len(got))
	}
}

// NewRenderPlan takes a caller-supplied (possibly deserialized) Template. A
// malformed one must yield an error, not a panic.
func TestNewRenderPlanRejectsMalformed(t *testing.T) {
	cases := []struct {
		name string
		tmpl Template
	}{
		{
			name: "nil params",
			tmpl: Template{Tokens: []string{"a"}, Params: nil, TokenCount: 1},
		},
		{
			name: "dense tokens shorter than non-param positions",
			tmpl: Template{Tokens: []string{"a"}, Params: bitset.New(2), TokenCount: 2},
		},
		{
			name: "stray param bit beyond token count",
			tmpl: func() Template {
				p := bitset.New(64)
				p.Set(40)
				return Template{Tokens: []string{"a", "b"}, Params: p, TokenCount: 3}
			}(),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewRenderPlan(tc.tmpl, nil); err == nil {
				t.Fatalf("expected error for malformed template, got nil")
			}
		})
	}
}
