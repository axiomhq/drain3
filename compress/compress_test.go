package compress

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

func TestCompressValueAt(t *testing.T) {
	lines := []string{
		"user alice logged in from 10.0.0.1",
		"user bob logged in from 10.0.0.2",
		"user charlie logged in from 192.168.1.1",
		"server started on port 8080",
		"server started on port 9090",
		"this is a completely unique snowflake line",
	}

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	for i, want := range lines {
		got := dc.ValueAt(i)
		if got != want {
			t.Errorf("ValueAt(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestCompressWriteToReadFrom(t *testing.T) {
	lines := []string{
		"user alice logged in from 10.0.0.1",
		"user bob logged in from 10.0.0.2",
		"user charlie logged in from 192.168.1.1",
		"server started on port 8080",
		"server started on port 9090",
		"this is a completely unique snowflake line",
	}

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	var buf bytes.Buffer
	n, err := dc.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	if n != int64(buf.Len()) {
		t.Fatalf("WriteTo returned %d, buffer has %d bytes", n, buf.Len())
	}
	t.Logf("serialized %d lines into %d bytes (raw %d bytes)", len(lines), n, rawSize(lines))

	dc2, err := ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}

	for i, want := range lines {
		got := dc2.ValueAt(i)
		if got != want {
			t.Errorf("round-trip ValueAt(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestCompressEmpty(t *testing.T) {
	dc, err := Compress(nil)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	var buf bytes.Buffer
	if _, err := dc.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	dc2, err := ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	_ = dc2
}

func TestCompressAllUnmatched(t *testing.T) {
	lines := []string{
		"alpha",
		"bravo charlie",
		"delta echo foxtrot golf",
	}

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	for i, want := range lines {
		got := dc.ValueAt(i)
		if got != want {
			t.Errorf("ValueAt(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestCompressManyLines(t *testing.T) {
	var lines []string
	for i := range 200 {
		lines = append(lines, fmt.Sprintf("request %d from client %d took %dms", i, i%10, i*3))
	}
	lines = append(lines, "shutdown initiated")
	lines = append(lines, "goodbye")

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	for i, want := range lines {
		got := dc.ValueAt(i)
		if got != want {
			t.Errorf("ValueAt(%d) = %q, want %q", i, got, want)
		}
	}

	// Round-trip through WriteTo/ReadFrom.
	var buf bytes.Buffer
	if _, err := dc.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	dc2, err := ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	for i, want := range lines {
		got := dc2.ValueAt(i)
		if got != want {
			t.Errorf("round-trip ValueAt(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestValueAtPanicsOutOfRange(t *testing.T) {
	dc, err := Compress([]string{"hello world"})
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-range index")
		}
	}()
	dc.ValueAt(5)
}

func TestDecompress(t *testing.T) {
	lines := []string{
		"user alice logged in from 10.0.0.1",
		"user bob logged in from 10.0.0.2",
		"user charlie logged in from 192.168.1.1",
		"server started on port 8080",
		"server started on port 9090",
		"this is a completely unique snowflake line",
	}

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	got := Decompress(dc)
	if len(got) != len(lines) {
		t.Fatalf("Decompress returned %d lines, want %d", len(got), len(lines))
	}
	for i, want := range lines {
		if got[i] != want {
			t.Errorf("Decompress[%d] = %q, want %q", i, got[i], want)
		}
	}
}

func TestDecompressRoundTrip(t *testing.T) {
	lines := benchLines()

	dc, err := Compress(lines)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	var buf bytes.Buffer
	if _, err := dc.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	dc2, err := ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}

	got := Decompress(dc2)
	if len(got) != len(lines) {
		t.Fatalf("Decompress returned %d lines, want %d", len(got), len(lines))
	}
	for i, want := range lines {
		if got[i] != want {
			t.Errorf("round-trip Decompress[%d] = %q, want %q", i, got[i], want)
		}
	}
}

func benchLines() []string {
	lines := make([]string, 0, 202)
	for i := range 200 {
		lines = append(lines, fmt.Sprintf("request %d from client %d took %dms", i, i%10, i*3))
	}
	lines = append(lines, "shutdown initiated")
	lines = append(lines, "goodbye")
	return lines
}

func BenchmarkCompress(b *testing.B) {
	lines := benchLines()
	b.ResetTimer()
	for range b.N {
		dc, err := Compress(lines)
		if err != nil {
			b.Fatal(err)
		}
		_ = dc
	}
}

func BenchmarkValueAt(b *testing.B) {
	lines := benchLines()
	dc, err := Compress(lines)
	if err != nil {
		b.Fatal(err)
	}
	n := dc.Len()
	b.ResetTimer()
	for i := range b.N {
		_ = dc.ValueAt(i % n)
	}
}

func BenchmarkWriteTo(b *testing.B) {
	lines := benchLines()
	dc, err := Compress(lines)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		dc.WriteTo(io.Discard)
	}
}

func BenchmarkReadFrom(b *testing.B) {
	lines := benchLines()
	dc, err := Compress(lines)
	if err != nil {
		b.Fatal(err)
	}
	var buf bytes.Buffer
	dc.WriteTo(&buf)
	data := buf.Bytes()
	b.ResetTimer()
	for range b.N {
		_, err := ReadFrom(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecompress(b *testing.B) {
	lines := benchLines()
	dc, err := Compress(lines)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		_ = Decompress(dc)
	}
}

func rawSize(lines []string) int {
	n := 0
	for _, l := range lines {
		n += len(l)
	}
	return n
}
