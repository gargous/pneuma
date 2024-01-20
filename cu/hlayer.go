package cu

import (
	"math"
	"pneuma/cnn"
	"pneuma/nn"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	*cnn.HLayerConv
	cal *MatCaltor
}

func NewHLayerConv(cal *MatCaltor, param cnn.ConvKernalParam) *HLayerConv {
	return &HLayerConv{
		HLayerConv: cnn.NewHLayerConv(param),
		cal:        cal,
	}
}

func (l *HLayerConv) InitSize(size []int) []int {
	ret := l.HLayerConv.InitSize(size)
	l.cal.CopyTo(l.W, l.DW, l.B, l.DB)
	return ret
}

func (l *HLayerConv) Predict(x *mat.Dense) (y *mat.Dense) {
	y = l.Forward(x)
	l.cal.Clear(l.PackX)
	l.PackX = nil
	return
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	batch := x.RawMatrix().Cols
	wr, _ := l.W.Dims()
	br, bc := l.B.Dims()
	packX := mat.NewDense(br*batch, wr, nil)
	l.C.FoldBatches(x, packX, l.C.PackTo)
	packY := mat.NewDense(br*batch, bc, nil)
	l.PackX = packX

	//gpu cal start
	l.cal.CopyTo(l.PackX, packY)
	l.cal.Mul(packY, l.PackX, l.W, false, false)
	l.cal.AddScaledRowByOne(packY, 1, l.B)
	l.cal.CopyBack(packY)
	l.cal.Clear(packY)
	//gpu cal end

	y = mat.NewDense(br*bc, batch, nil)
	l.C.UnfoldBatches(y, packY, nil)
	return
}

func (l *HLayerConv) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	wr, _ := l.W.Dims()
	br, bc := l.B.Dims()
	l.DW.Zero()
	l.DB.Zero()
	packDy := mat.NewDense(br*batch, bc, nil)
	l.C.FoldBatches(dy, packDy, nil)
	packDx := mat.NewDense(br*batch, wr, nil)

	//gpu cal start
	l.cal.CopyTo(packDx, packDy, l.DW, l.DB)
	l.cal.Mul(l.DW, l.PackX, packDy, true, false)
	l.cal.AddScaledOneByRow(l.DB, 1, packDy)
	l.cal.Mul(packDx, packDy, l.W, false, true)
	l.cal.CopyBack(packDx)
	l.cal.Clear(packDx, packDy, l.PackX)
	//gpu cal end

	_, orgSizeSum := l.C.OrgSize()
	dx = mat.NewDense(orgSizeSum, batch, nil)
	l.C.UnfoldBatches(dx, packDx, l.C.UnPackTo)
	return
}

type HLayerBatchNorm struct {
	*nn.HLayerBatchNorm
	cal *MatCaltor
	S   *mat.VecDense
}

func NewHLayerBatchNorm(cal *MatCaltor, minstd, momentum float64) *HLayerBatchNorm {
	return &HLayerBatchNorm{
		HLayerBatchNorm: nn.NewHLayerBatchNorm(minstd, momentum),
		cal:             cal,
	}
}

func (l *HLayerBatchNorm) InitSize(size []int) []int {
	l.HLayerBatchNorm.InitSize(size)
	r := size[0]
	l.DB = mat.NewVecDense(r, nil)
	l.DG = mat.NewVecDense(r, nil)
	l.S = mat.NewVecDense(r, nil)
	l.cal.CopyTo(l.E, l.S, l.B, l.DB)
	return size
}

func (l *HLayerBatchNorm) forward(xsube *mat.Dense, s *mat.VecDense) (y *mat.Dense, sInverse *mat.VecDense) {
	r, c := xsube.Dims()
	l.cal.CopyBack(s)
	sInverse = mat.NewVecDense(s.Len(), nil)
	alpha := 1.0 / math.Sqrt(float64(c))
	for i := 0; i < r; i++ {
		sInverse.SetVec(i, alpha/(s.AtVec(i)+l.MinStd))
	}
	y = mat.NewDense(r, c, nil)
	l.cal.CopyTo(y)
	l.cal.MulElemColByOneHost(xsube, sInverse)
	l.cal.CopyInDevice(y, xsube)
	l.cal.MulElemColByOneHost(y, l.G)
	l.cal.AddScaledColByOne(y, 1, l.B)
	return
}

func (l *HLayerBatchNorm) Predict(x *mat.Dense) (y *mat.Dense) {
	l.cal.CopyTo(x)
	l.cal.AddScaledColByOne(x, -1, l.E)
	y, l.SInverse = l.forward(x, l.S)
	l.cal.CopyBack(y)
	l.cal.Clear(y, x)
	return
}

func (l *HLayerBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	r, c := x.Dims()
	e := mat.NewVecDense(r, nil)
	s := mat.NewVecDense(r, nil)
	ones := mat.NewVecDense(r, nil)
	alpha := 1.0 / float64(c)
	floats.AddConst(alpha, ones.RawVector().Data)
	l.cal.CopyTo(e, s, x, ones)
	l.cal.Mul(e, x, ones, false, false)
	l.cal.AddScaledColByOne(x, -1, e)
	l.cal.NormOneByRow(s, x)
	l.cal.Scale(alpha, s)
	l.cal.Scale(1-l.Momentum, l.E)
	l.cal.AddScaled(l.E, l.Momentum, e)
	l.cal.Scale(1-l.Momentum, l.S)
	l.cal.AddScaled(l.S, l.Momentum, s)
	y, l.SInverse = l.forward(x, s)
	l.XHat = x
	l.cal.CopyBack(y)
	l.cal.Clear(e, s, y, ones)
	return
}

func (l *HLayerBatchNorm) Backward(dy *mat.Dense) (dx *mat.Dense) {
	_, xc := dy.Dims()
	m := float64(xc)
	dx = mat.DenseCopyOf(dy)
	l.DB.Zero()
	l.DG.Zero()
	l.cal.CopyTo(dx, l.DB)
	// dy <=> dx
	//sum(xhat*dy) <=> dg
	sumXhatDy := l.DG
	//d2 = xhat = xhat*sum(xhat*dy)
	l.cal.DotRowByRowToHost(sumXhatDy, l.XHat, dx)
	l.cal.MulElemColByOneHost(l.XHat, sumXhatDy)
	d2 := l.XHat
	//sum(dy) <=> db
	d3 := l.DB
	//d3 = sum(dy)
	l.cal.AddScaledOneByCol(d3, 1, dx)
	//d1 <=> dx
	d1 := dx
	//d1 = dx = m * dx = m* dy
	l.cal.Scale(m, d1)
	//scaler = g * si / m
	l.SInverse.MulElemVec(l.SInverse, l.G)
	l.SInverse.ScaleVec(1.0/m, l.SInverse)
	scaler := l.SInverse
	//dx = scaler*d1 = scaler(d1-d2-d3)
	l.cal.AddScaled(d1, -1, d2)
	l.cal.AddScaledColByOne(d1, -1, d3)
	l.cal.MulElemColByOneHost(d1, scaler)

	l.cal.CopyBack(dx)
	l.cal.Clear(dx)
	return
}

type HLayerDimBatchNorm struct {
	*HLayerBatchNorm
	picker   *cnn.MatPicker
	batchDim int
}

func NewHLayerConvBatchNorm(cal *MatCaltor, minstd, momentum float64) *HLayerDimBatchNorm {
	return &HLayerDimBatchNorm{
		batchDim:        -1,
		HLayerBatchNorm: NewHLayerBatchNorm(cal, minstd, momentum),
	}
}

func (l *HLayerDimBatchNorm) InitSize(size []int) []int {
	batchDim := (l.batchDim + len(size)) % len(size)
	l.picker = cnn.NewMatPicker(size, batchDim)
	l.HLayerBatchNorm.InitSize([]int{size[batchDim]})
	return size
}

func (l *HLayerDimBatchNorm) Predict(x *mat.Dense) (y *mat.Dense) {
	l.picker.SetMotion(0, 1)
	newX := l.picker.Pick(x)
	newY := l.HLayerBatchNorm.Predict(newX)
	l.picker.SetMotion(1, 0)
	return l.picker.Pick(newY)
}

func (l *HLayerDimBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	l.picker.SetMotion(0, 1)
	newX := l.picker.Pick(x)
	newY := l.HLayerBatchNorm.Forward(newX)
	l.picker.SetMotion(1, 0)
	return l.picker.Pick(newY)
}

func (l *HLayerDimBatchNorm) Backward(dy *mat.Dense) (dx *mat.Dense) {
	l.picker.SetMotion(0, 1)
	newDy := l.picker.Pick(dy)
	newDx := l.HLayerBatchNorm.Backward(newDy)
	l.picker.SetMotion(1, 0)
	return l.picker.Pick(newDx)
}
