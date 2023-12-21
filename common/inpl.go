package common

import "gonum.org/v1/gonum/mat"

type DenseCaltorNorm struct {
}

func NewDenseCaltorNorm() *DenseCaltorNorm {
	return &DenseCaltorNorm{}
}

func (d *DenseCaltorNorm) Mul(dst, a, b *mat.Dense) {
	dst.Mul(a, b)
}
