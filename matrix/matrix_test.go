package matrix

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func assertEqualMatrixes(t *testing.T, a, b mat.Matrix) {
	if !mat.Equal(a, b) {
		fa := mat.Formatted(a, mat.Prefix(""), mat.Squeeze())
		fb := mat.Formatted(b, mat.Prefix(""), mat.Squeeze())
		t.Errorf("\ngot:\n\n%v\n\nwanted:\n\n%v", fa, fb)
	}
}

func TestAdd(t *testing.T) {
	want := mat.NewDense(2, 2, []float64{5, 6, 2, 3})
	a := mat.NewDense(2, 2, []float64{2, 1, 0, 2})
	b := mat.NewDense(2, 2, []float64{3, 5, 2, 1})
	got := Add(a, b)

	assertEqualMatrixes(t, got, want)
}

func TestAddScalar(t *testing.T) {
	want := mat.NewDense(2, 1, []float64{2, 4})
	a := mat.NewDense(2, 1, []float64{1, 3})
	got := AddScalar(a, 1)

	assertEqualMatrixes(t, got, want)
}

func TestApply(t *testing.T) {
	want := mat.NewDense(2, 2, []float64{4, 25, 16, 9})
	square := func(r, c int, f float64) float64 { return math.Pow(f, 2) }
	a := mat.NewDense(2, 2, []float64{2, 5, 4, 3})
	got := Apply(square, a)

	assertEqualMatrixes(t, got, want)
}

func TestDot(t *testing.T) {
	want := mat.NewDense(2, 1, []float64{11, 4})
	a := mat.NewDense(2, 2, []float64{1, 2, 0, 1})
	b := mat.NewDense(2, 1, []float64{3, 4})
	got, err := Dot(a, b)
	if err != nil {
		t.Errorf("Error when testing Dot(): %v", err)
	}

	assertEqualMatrixes(t, got, want)
}

func TestMultiply(t *testing.T) {
	want := mat.NewDense(2, 2, []float64{12, 4, 36, 50})
	a := mat.NewDense(2, 2, []float64{3, 2, 9, 2})
	b := mat.NewDense(2, 2, []float64{4, 2, 4, 25})
	got := Multiply(a, b)

	assertEqualMatrixes(t, got, want)
}

func TestScale(t *testing.T) {
	want := mat.NewDense(2, 2, []float64{2, 4, 4, 6})
	a := mat.NewDense(2, 2, []float64{1, 2, 2, 3})
	got := Scale(a, 2)

	assertEqualMatrixes(t, got, want)
}

func TestSubstract(t *testing.T) {
	want := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	a := mat.NewDense(2, 2, []float64{5, 3, 6, 4})
	b := mat.NewDense(2, 2, []float64{4, 1, 3, 0})
	got := Substract(a, b)

	assertEqualMatrixes(t, got, want)
}
