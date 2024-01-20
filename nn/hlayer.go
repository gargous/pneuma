package nn

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type HLayerLinear struct {
	w  *mat.Dense
	b  *mat.VecDense
	dw *mat.Dense
	db *mat.VecDense
	x  *mat.Dense
	Y  *mat.Dense
}

func NewHLayerLinear() *HLayerLinear {
	return &HLayerLinear{}
}

func (l *HLayerLinear) InitSize(size []int) []int {
	r, c := size[0], size[1]
	l.w = mat.NewDense(r, c, nil)
	l.b = mat.NewVecDense(r, nil)
	l.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, l.w)
	return size
}

func (l *HLayerLinear) Dims() (r, c int) {
	return l.w.Dims()
}

func (l *HLayerLinear) Forward(x *mat.Dense) (y *mat.Dense) {
	_, c := x.Dims()
	r := l.b.Len()
	y = mat.NewDense(r, c, nil)
	y.Mul(l.w, x)
	for j := 0; j < c; j++ {
		yCol := mat.NewVecDense(r, nil)
		yCol.AddVec(y.ColView(j), l.b)
		y.SetCol(j, yCol.RawVector().Data)
	}
	l.x = x
	l.Y = y
	return
}

func (l *HLayerLinear) Backward(dy *mat.Dense) (dx *mat.Dense) {
	wr, wc := l.w.Dims()
	xr, xc := l.x.Dims()
	db := mat.NewVecDense(wr, nil)
	dw := mat.NewDense(wr, wc, nil)
	for j := 0; j < xc; j++ {
		db.AddVec(db, dy.ColView(j))
	}
	dw.Mul(dy, l.x.T())
	l.dw = dw
	l.db = db
	dx = mat.NewDense(xr, xc, nil)
	dx.Mul(l.w.T(), dy)
	return
}

func (l *HLayerLinear) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.w, l.b,
	}
	deltas = []mat.Matrix{
		l.dw, l.db,
	}
	return
}

type HLayerBatchNorm struct {
	E        *mat.VecDense
	V        *mat.VecDense
	SInverse *mat.VecDense
	G        *mat.VecDense
	B        *mat.VecDense
	DG       *mat.VecDense
	DB       *mat.VecDense
	XHat     *mat.Dense
	MinStd   float64
	Momentum float64
}

func NewHLayerBatchNorm(minstd, momentum float64) *HLayerBatchNorm {
	l := &HLayerBatchNorm{}
	l.MinStd = minstd
	l.Momentum = momentum
	return l
}

func (l *HLayerBatchNorm) forward(xsube *mat.Dense, v *mat.VecDense) (y, xhat *mat.Dense) {
	r, c := xsube.Dims()
	for i := 0; i < r; i++ {
		l.SInverse.SetVec(i, 1.0/math.Sqrt(v.AtVec(i)+l.MinStd))
	}
	y = mat.NewDense(r, c, nil)
	xhat = mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		xsubeCol := xsube.ColView(j)
		xhatCol := xhat.ColView(j).(*mat.VecDense)
		xhatCol.MulElemVec(xsubeCol, l.SInverse)
		yCol := y.ColView(j).(*mat.VecDense)
		yCol.MulElemVec(xhatCol, l.G)
		yCol.AddVec(yCol, l.B)
	}
	return
}

func (l *HLayerBatchNorm) InitSize(size []int) []int {
	r := size[0]
	l.E = mat.NewVecDense(r, nil)
	l.V = mat.NewVecDense(r, nil)
	l.G = mat.NewVecDense(r, nil)
	floats.AddConst(1, l.G.RawVector().Data)
	l.B = mat.NewVecDense(r, nil)
	l.SInverse = mat.NewVecDense(r, nil)
	return size
}

func (l *HLayerBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	e := mat.NewVecDense(r, nil)
	v := mat.NewVecDense(r, nil)
	ones := mat.NewVecDense(c, nil)
	alpha := 1.0 / float64(c)
	floats.AddConst(alpha, ones.RawVector().Data)
	e.MulVec(x, ones)
	xsube := mat.NewDense(r, c, nil)
	xsubeSqual := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		eCol := e
		xCol := x.ColView(j)
		xsubeCol := xsube.ColView(j).(*mat.VecDense)
		xsubeCol.SubVec(xCol, eCol)
	}
	xsubeSqual.MulElem(xsube, xsube)
	v.MulVec(xsubeSqual, ones)
	l.E.ScaleVec(1-l.Momentum, l.E)
	l.E.AddScaledVec(l.E, l.Momentum, e)
	l.V.ScaleVec(1-l.Momentum, l.V)
	l.V.AddScaledVec(l.V, l.Momentum, v)
	y, l.XHat = l.forward(xsube, v)
	return
}

func (l *HLayerBatchNorm) Predict(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	e := l.E
	v := l.V
	xsube := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		eCol := e
		xCol := x.ColView(j)
		xsubeCol := xsube.ColView(j).(*mat.VecDense)
		xsubeCol.SubVec(xCol, eCol)
	}
	y, _ = l.forward(xsube, v)
	return
}

func (l *HLayerBatchNorm) Backward(dy *mat.Dense) (dx *mat.Dense) {
	xr, xc := dy.Dims()
	dx = mat.NewDense(xr, xc, nil)
	m := float64(xc)
	dg := mat.NewVecDense(xr, nil)
	db := mat.NewVecDense(xr, nil)
	for i := 0; i < xr; i++ {
		dyRow := dy.RowView(i)
		sumDyRow := mat.Sum(dyRow)
		xhatRow := l.XHat.RowView(i)
		sumXhatDyRow := mat.Dot(dyRow, xhatRow)
		si := l.SInverse.AtVec(i)
		//scaler = g * si / m
		scaler := l.G.AtVec(i) * si / m
		//d1 = m * dy
		d1 := mat.VecDenseCopyOf(dyRow)
		d1.ScaleVec(m, d1)
		//d2 = xhat*sum(hat*dy)
		d2 := mat.VecDenseCopyOf(xhatRow)
		d2.ScaleVec(sumXhatDyRow, d2)
		//d3 = sum(dy)
		d3 := mat.NewVecDense(xc, nil)
		floats.AddConst(sumDyRow, d3.RawVector().Data)
		//dx = scaler*(d1-d2-d3)
		dxCol := d1
		dxCol.SubVec(dxCol, d2)
		dxCol.SubVec(dxCol, d3)
		dxCol.ScaleVec(scaler, dxCol)
		dx.SetRow(i, dxCol.RawVector().Data)
		//dg = sum(hat*dy)
		dg.SetVec(i, sumXhatDyRow)
		//db = sum(dy)
		db.SetVec(i, sumDyRow)
	}
	l.DG = dg
	l.DB = db
	return
}

func (l *HLayerBatchNorm) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.G, l.B,
	}
	deltas = []mat.Matrix{
		l.DG, l.DB,
	}
	return
}

type HLayerSigmoid struct {
	y *mat.Dense
}

func NewHLayerSigmoid() *HLayerSigmoid {
	return &HLayerSigmoid{}
}

func (l *HLayerSigmoid) Forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	y = mat.NewDense(r, c, nil)
	y.Apply(func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}, x)
	l.y = y
	return
}

func (l *HLayerSigmoid) Backward(dy *mat.Dense) (dx *mat.Dense) {
	r, c := dy.Dims()
	one := mat.NewDense(r, c, nil)
	floats.AddConst(1, one.RawMatrix().Data)
	oneSubY := one
	oneSubY.Sub(one, l.y)
	dx = oneSubY
	dx.MulElem(l.y, oneSubY)
	dx.MulElem(dx, dy)
	return
}

type HLayerRelu struct {
	phi *mat.Dense
}

func NewHLayerRelu() *HLayerRelu {
	return &HLayerRelu{}
}

func (l *HLayerRelu) Forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	phi := mat.NewDense(r, c, nil)
	phi.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, x)
	l.phi = phi
	y = mat.NewDense(r, c, nil)
	y.MulElem(phi, x)
	return
}

func (l *HLayerRelu) Backward(dy *mat.Dense) (dx *mat.Dense) {
	r, c := dy.Dims()
	dx = mat.NewDense(r, c, nil)
	dx.MulElem(l.phi, dy)
	return
}
