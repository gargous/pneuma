package cu

import (
	"pneuma/fecs"

	"gonum.org/v1/gonum/mat"
)

type PCA struct {
	*fecs.PCA
	c *MatCaltor
}

func NewPCA(c *MatCaltor) *PCA {
	return &PCA{c: c, PCA: fecs.NewPCA()}
}

func (p *PCA) covMatEIG(covMat mat.Matrix, eigLen int) [][]float64 {
	covLen, _ := covMat.Dims()
	svdS := mat.NewVecDense(covLen, nil)
	svdU := mat.NewDense(covLen, covLen, nil)
	svdV := mat.NewDense(covLen, covLen, nil)
	p.c.CopyTo(svdS, svdU, svdV)
	p.c.SVD(covMat, svdU, svdS, svdV)
	p.c.CopyBack(svdS, svdU)
	p.c.Clear(svdS, svdU, svdV)
	return p.PickEIG(svdU, svdS.RawVector().Data, eigLen)
}

func (p *PCA) ColMod(dst *mat.Dense, src mat.Matrix) {
	r, c := src.Dims()
	k, _ := dst.Dims()
	e := mat.NewVecDense(r, nil)
	alpha := 1.0 / float64(c)
	xsube := mat.DenseCopyOf(src)
	p.c.CopyTo(xsube, e)
	p.c.AddScaledOneByCol(e, alpha, src)
	p.c.AddScaledColByOne(xsube, -1, e)

	covMat := mat.NewDense(r, r, nil)
	p.c.CopyTo(covMat)
	p.c.Mul(covMat, xsube, xsube, false, true)
	p.c.Scale(alpha, covMat)
	eigMat := mat.NewDense(k, r, nil)
	eigVecs := p.covMatEIG(covMat, k)
	for i := 0; i < k; i++ {
		eigMat.SetRow(i, eigVecs[i])
	}
	p.c.CopyTo(eigMat)
	p.c.Mul(dst, eigMat, src, false, false)
	p.c.Clear(eigMat, covMat)
}

func (p *PCA) RowMod(dst *mat.Dense, src mat.Matrix) {
	r, c := src.Dims()
	_, k := dst.Dims()
	e := mat.NewVecDense(c, nil)
	alpha := 1.0 / float64(r)
	xsube := mat.DenseCopyOf(src)
	p.c.CopyTo(xsube, e)
	p.c.AddScaledOneByRow(e, alpha, src)
	p.c.AddScaledRowByOne(xsube, -1, e)

	covMat := mat.NewDense(c, c, nil)
	p.c.CopyTo(covMat)
	p.c.Mul(covMat, xsube, xsube, true, false)
	p.c.Scale(alpha, covMat)
	eigMat := mat.NewDense(c, k, nil)
	eigVecs := p.covMatEIG(covMat, k)
	for j := 0; j < k; j++ {
		eigMat.SetCol(j, eigVecs[j])
	}
	p.c.CopyTo(eigMat)
	p.c.Mul(dst, src, eigMat, false, false)
	p.c.Clear(eigMat, covMat)
}
