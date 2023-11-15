package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	param *Param
}

func (l *Layer) forward(input *mat.Dense) {

}

type LinearLayer struct {
	w  *mat.Dense
	b  *mat.VecDense
	dw *mat.Dense
	db *mat.VecDense
	x  *mat.Dense
}

func (l *LinearLayer) forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	y = mat.NewDense(l.b.Len(), c, nil)
	y.Mul(l.w, x)
	for j := 0; j < c; j++ {
		yCol := mat.NewVecDense(r, nil)
		yCol.AddVec(y.ColView(j), l.b)
		y.SetCol(j, yCol.RawVector().Data)
	}
	l.x = x
	return
}

func (l *LinearLayer) backward(dy *mat.Dense) (dx *mat.Dense) {
	wr, wc := l.w.Dims()
	xr, xc := l.x.Dims()
	db := mat.NewVecDense(wr, nil)
	dw := mat.NewDense(wr, wc, nil)
	for j := 0; j < wc; j++ {
		db.AddVec(db, dy.ColView(j))
	}
	dw.Mul(dy, l.x.T())
	l.dw = dw
	l.db = db
	dx = mat.NewDense(xc, xr, nil)
	dx.Mul(dy.T(), l.w)
	dx = dx.T().(*mat.Dense)
	return
}

type BatchNormLayer struct {
	e        *mat.VecDense
	v        *mat.VecDense
	sinverse *mat.VecDense
	g        *mat.VecDense
	b        *mat.VecDense
	dg       *mat.VecDense
	db       *mat.VecDense
	xhat     *mat.VecDense
	minstd   float64
	momentum float64
}

func (l *BatchNormLayer) forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()

	e := mat.NewVecDense(r, nil)
	v := mat.NewVecDense(r, nil)
	for j := 0; j < c; j++ {
		xColData := mat.Col(nil, j, x)
		e.AddVec(e, mat.NewVecDense(len(xColData), xColData))
	}
	e.ScaleVec(1.0/float64(c), e)
	for j := 0; j < c; j++ {
		eCol := e
		xCol := mat.NewVecDense(eCol.Len(), mat.Col(nil, j, x))
		vCol := mat.NewVecDense(eCol.Len(), nil)
		vCol.SubVec(xCol, eCol)
		vCol.MulElemVec(vCol, vCol)
		v.AddVec(v, vCol)
	}
	v.ScaleVec(1.0/float64(c), v)

	l.e.ScaleVec(1-l.momentum, l.e)
	l.e.AddScaledVec(l.e, l.momentum, e)
	l.v.ScaleVec(1-l.momentum, l.v)
	l.v.AddScaledVec(l.v, l.momentum, v)

	sinverse := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		sinverse.SetVec(i, 1.0/math.Sqrt(v.AtVec(i)+l.minstd))
	}
	y = mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		xCol := mat.NewVecDense(e.Len(), mat.Col(nil, j, x))
		xHatCol := mat.NewVecDense(e.Len(), nil)
		xHatCol.SubVec(xCol, e)
		xHatCol.MulElemVec(xHatCol, sinverse)
		yCol := xHatCol
		yCol.MulElemVec(yCol, l.g)
		yCol.AddVec(yCol, l.b)
		y.SetCol(j, yCol.RawVector().Data)
	}
	return
}

func (l *BatchNormLayer) backward(x *mat.Dense) (d *mat.Dense) {

}
