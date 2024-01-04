package fecs

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPCA(t *testing.T) {
	src := mat.NewDense(2, 5, []float64{
		-1, -1, 0, 2, 0,
		-2, 0, 0, 1, 1,
	})
	dst := mat.NewDense(1, 5, nil)
	pca := NewPCA()
	pca.ColMod(dst, src)
	tar := mat.NewDense(1, 5, []float64{
		2.12132, 0.707107, 0, -2.12132, -0.707107,
	})
	if !mat.EqualApprox(dst, tar, 0.000001) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
}
