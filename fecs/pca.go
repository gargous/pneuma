package fecs

import (
	"sort"

	"gonum.org/v1/gonum/mat"
)

type PCA struct {
}

func NewPCA() *PCA {
	return &PCA{}
}

func (p *PCA) PickEIG(u *mat.Dense, s []float64, eigLen int) [][]float64 {
	size, _ := u.Dims()
	eigVecs := make([][][]float64, size)
	for i := 0; i < size; i++ {
		eigVecs[i] = [][]float64{
			u.RawRowView(i),
			{s[i]},
		}
	}
	sort.Slice(eigVecs, func(i, j int) bool {
		return eigVecs[i][1][0] > eigVecs[j][1][0]
	})
	ret := make([][]float64, eigLen)
	for i := 0; i < eigLen; i++ {
		ret[i] = eigVecs[i][0]
	}
	return ret
}

func (p *PCA) covMatEIG(covMat mat.Matrix, eigLen int) [][]float64 {
	covLen, _ := covMat.Dims()
	svdU := mat.NewDense(covLen, covLen, nil)
	svdSolver := mat.SVD{}
	svdSolver.Factorize(covMat, mat.SVDFull)
	svdS := svdSolver.Values(nil)
	svdSolver.UTo(svdU)
	return p.PickEIG(svdU, svdS, eigLen)
}

// feature in col
func (p *PCA) ColMod(dst *mat.Dense, src mat.Matrix) {
	r, c := src.Dims()
	k, _ := dst.Dims()
	e := mat.NewVecDense(r, nil)
	alpha := 1.0 / float64(c)
	xsube := mat.DenseCopyOf(src)
	for j := 0; j < c; j++ {
		e.AddScaledVec(e, alpha, xsube.ColView(j))
	}
	for j := 0; j < c; j++ {
		xsubeCol := xsube.ColView(j).(*mat.VecDense)
		xsubeCol.AddScaledVec(xsubeCol, -1, e)
	}
	covMat := mat.NewDense(r, r, nil)
	covMat.Mul(xsube, xsube.T())
	covMat.Scale(alpha, covMat)
	eigVecs := p.covMatEIG(covMat, k)
	eigMat := mat.NewDense(k, r, nil)
	for i := 0; i < k; i++ {
		eigMat.SetRow(i, eigVecs[i])
	}
	dst.Mul(eigMat, src)
}

// feature in row
func (p *PCA) RowMod(dst *mat.Dense, src mat.Matrix) {
	r, c := src.Dims()
	_, k := dst.Dims()
	e := mat.NewVecDense(c, nil)
	alpha := 1.0 / float64(r)
	xsube := mat.DenseCopyOf(src)
	for i := 0; i < r; i++ {
		e.AddScaledVec(e, alpha, xsube.RowView(i))
	}
	for i := 0; i < r; i++ {
		xsubeCol := xsube.RowView(i).(*mat.VecDense)
		xsubeCol.AddScaledVec(xsubeCol, -1, e)
	}
	covMat := mat.NewDense(c, c, nil)
	covMat.Mul(xsube.T(), xsube)
	covMat.Scale(alpha, covMat)
	eigVecs := p.covMatEIG(covMat, k)
	eigMat := mat.NewDense(c, k, nil)
	for j := 0; j < k; j++ {
		eigMat.SetCol(j, eigVecs[j])
	}
	dst.Mul(src, eigMat)
}
