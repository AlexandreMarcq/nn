package matrix

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func checkSameShape(a, b mat.Matrix) error {
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	if ra != rb && ca != cb {
		return errors.New(fmt.Sprintf("matrices are not of the same shape: (%d, %d) and (%d, %d)", ra, ca, rb, cb))
	}
	return nil
}

func checkDotCompatible(a, b mat.Matrix) error {
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	if ca != rb {
		return errors.New(fmt.Sprintf("matrices are not dot compatible: (%d, %d) and (%d, %d)", ra, ca, rb, cb))
	}
	return nil
}

func Add(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	m := mat.NewDense(r, c, nil)
	m.Add(a, b)
	return m
}

func AddScalar(a mat.Matrix, f float64) mat.Matrix {
	r, c := a.Dims()
	s := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			s[c*i+j] = a.At(i, j) + f
		}
	}
	m := mat.NewDense(r, c, s)
	return m
}

func Apply(fn func(r, c int, f float64) float64, a mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	m := mat.NewDense(r, c, nil)
	m.Apply(fn, a)
	return m
}

func Dot(a, b mat.Matrix) (mat.Matrix, error) {
	err := checkDotCompatible(a, b)
	if err != nil {
		return nil, err
	}
	r, _ := a.Dims()
	_, c := b.Dims()
	m := mat.NewDense(r, c, nil)
	m.Product(a, b)
	return m, nil
}

func Multiply(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	m := mat.NewDense(r, c, nil)
	m.MulElem(a, b)
	return m
}

func Scale(a mat.Matrix, f float64) mat.Matrix {
	r, c := a.Dims()
	m := mat.NewDense(r, c, nil)
	m.Scale(f, a)
	return m
}

func Substract(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	m := mat.NewDense(r, c, nil)
	m.Sub(a, b)
	return m
}
