package cu

import (
	"pneuma/cnn"

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
	l.cal.Start(l.W, l.DW, l.B, l.DB)
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
	l.C.FoldBatches(x, packX, l.C.Pack)
	packY := mat.NewDense(br*batch, bc, nil)
	l.PackX = packX

	//gpu cal start
	l.cal.Start(l.PackX, packY)
	l.cal.Mul(packY, l.PackX, l.W, false, false)
	blen := br * bc
	for j := 0; j < len(packY.RawMatrix().Data); j += blen {
		l.cal.AddSlice(packY, l.B, j, 0, blen)
	}
	l.cal.End(packY)
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
	l.cal.Start(packDx, packDy)
	l.cal.Mul(l.DW, l.PackX, packDy, true, false)
	blen := br * bc
	for j := 0; j < len(packDy.RawMatrix().Data); j += blen {
		l.cal.AddSlice(l.DB, packDy, 0, j, blen)
	}
	l.cal.Mul(packDx, packDy, l.W, false, true)
	l.cal.End(packDx)
	l.cal.Clear(packDx, packDy, l.PackX)
	//gpu cal end

	_, orgSizeSum := l.C.OrgSize()
	dx = mat.NewDense(orgSizeSum, batch, nil)
	l.C.UnfoldBatches(dx, packDx, l.C.UnPack)
	return
}
