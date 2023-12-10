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
}

func NewHLayerLinear(r, c int) *HLayerLinear {
	l := &HLayerLinear{}
	l.w = mat.NewDense(r, c, nil)
	l.b = mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			l.w.Set(i, j, rand.Float64())
		}
		l.b.SetVec(i, rand.Float64())
	}
	return l
}

func (l *HLayerLinear) Copy(src *HLayerLinear) {
	l.w = mat.DenseCopyOf(src.w)
	l.b = mat.VecDenseCopyOf(src.b)
	if src.dw != nil {
		l.dw = mat.DenseCopyOf(src.dw)
		l.db = mat.VecDenseCopyOf(src.db)
	}
	if src.x != nil {
		l.x = mat.DenseCopyOf(src.x)
	}
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
	e        *mat.VecDense
	v        *mat.VecDense
	sinverse *mat.VecDense
	g        *mat.VecDense
	b        *mat.VecDense
	dg       *mat.VecDense
	db       *mat.VecDense
	xhat     *mat.Dense
	minstd   float64
	momentum float64
}

func NewHLayerBatchNorm(minstd, momentum float64) *HLayerBatchNorm {
	l := &HLayerBatchNorm{}
	l.minstd = minstd
	l.momentum = momentum
	return l
}

func (l *HLayerBatchNorm) Copy(src *HLayerBatchNorm) {
	l.e = mat.VecDenseCopyOf(src.e)
	l.v = mat.VecDenseCopyOf(src.v)
	l.g = mat.VecDenseCopyOf(src.g)
	l.b = mat.VecDenseCopyOf(src.b)
	l.minstd = src.minstd
	l.momentum = src.momentum
	if src.dg != nil {
		l.dg = mat.VecDenseCopyOf(src.dg)
		l.db = mat.VecDenseCopyOf(src.db)
	}
	if src.sinverse != nil {
		l.xhat = mat.DenseCopyOf(src.xhat)
		l.sinverse = mat.VecDenseCopyOf(src.sinverse)
	}
}

func (l *HLayerBatchNorm) forward(x, xsube *mat.Dense, e, v *mat.VecDense) (y, xhat *mat.Dense, sinverse *mat.VecDense) {
	r, c := x.Dims()
	sinverse = mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		sinverse.SetVec(i, 1.0/math.Sqrt(v.AtVec(i)+l.minstd))
	}
	y = mat.NewDense(r, c, nil)
	xhat = mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		xSubeCol := xsube.ColView(j)
		xHatCol := mat.NewVecDense(r, nil)
		xHatCol.MulElemVec(xSubeCol, sinverse)
		xhat.SetCol(j, xHatCol.RawVector().Data)

		yCol := mat.VecDenseCopyOf(xHatCol)
		yCol.MulElemVec(yCol, l.g)
		yCol.AddVec(yCol, l.b)
		y.SetCol(j, yCol.RawVector().Data)
	}
	return
}

func (l *HLayerBatchNorm) initOnData(x *mat.Dense) {
	if l.e != nil {
		return
	}
	r, _ := x.Dims()
	l.e = mat.NewVecDense(r, nil)
	l.v = mat.NewVecDense(r, nil)
	l.g = mat.NewVecDense(r, nil)
	floats.AddConst(1, l.g.RawVector().Data)
	l.b = mat.NewVecDense(r, nil)
}

func (l *HLayerBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	l.initOnData(x)

	e := mat.NewVecDense(r, nil)
	v := mat.NewVecDense(r, nil)
	for j := 0; j < c; j++ {
		e.AddVec(e, x.ColView(j))
	}
	e.ScaleVec(1.0/float64(c), e)
	xsube := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		eCol := e
		xCol := x.ColView(j)

		xsubeCol := mat.NewVecDense(r, nil)
		xsubeCol.SubVec(xCol, eCol)
		xsube.SetCol(j, xsubeCol.RawVector().Data)

		vCol := mat.NewVecDense(r, nil)
		vCol.MulElemVec(xsubeCol, xsubeCol)
		v.AddVec(v, vCol)
	}
	v.ScaleVec(1.0/float64(c), v)

	l.e.ScaleVec(1-l.momentum, l.e)
	l.e.AddScaledVec(l.e, l.momentum, e)
	l.v.ScaleVec(1-l.momentum, l.v)
	l.v.AddScaledVec(l.v, l.momentum, v)

	y, l.xhat, l.sinverse = l.forward(x, xsube, e, v)
	return
}

func (l *HLayerBatchNorm) Predict(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	e := l.e
	v := l.v
	xsube := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		eCol := e
		xCol := x.ColView(j)

		xsubeCol := mat.NewVecDense(r, nil)
		xsubeCol.SubVec(xCol, eCol)
		xsube.SetCol(j, xsubeCol.RawVector().Data)
	}
	y, _, _ = l.forward(x, xsube, e, v)
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
		xhatRow := l.xhat.RowView(i)
		sumXhatDyRow := mat.Dot(dyRow, xhatRow)
		si := l.sinverse.AtVec(i)
		//scaler = g * si / m
		scaler := l.g.AtVec(i) * si / m
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
	l.dg = dg
	l.db = db
	return
}

func (l *HLayerBatchNorm) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.g, l.b,
	}
	deltas = []mat.Matrix{
		l.dg, l.db,
	}
	return
}

type HLayerSigmoid struct {
	y *mat.Dense
}

func NewHLayerSigmoid() *HLayerSigmoid {
	return &HLayerSigmoid{}
}

func (l *HLayerSigmoid) Copy(src *HLayerSigmoid) {
	if src.y != nil {
		l.y = &mat.Dense{}
		l.y.CloneFrom(src.y)
	}
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

func (l *HLayerRelu) Copy(src *HLayerRelu) {
	if src.phi != nil {
		l.phi = &mat.Dense{}
		l.phi.CloneFrom(src.phi)
	}
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
