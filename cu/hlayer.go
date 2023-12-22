package cu

import (
	"math/rand"
	"pneuma/cnn"

	"gonum.org/v1/gonum/mat"
)

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	*MatCaltor
	c        *cnn.ConvPacker
	w        *mat.Dense
	b        *mat.Dense
	dw       *mat.Dense
	db       *mat.Dense
	packX    *mat.Dense
	ouptSize []int
}

func NewHLayerConv(e *Engine, inputSize, core, stride []int, padding bool) *HLayerConv {
	coreCnt := core[len(core)-1]
	ret := &HLayerConv{}
	ret.c = cnn.NewConvPacker(
		inputSize,
		core[:len(core)-1],
		stride,
		padding,
	)
	slipCnt, slipCntSum := ret.c.SlipCnt()
	_, coreSizeSum := ret.c.CoreSize()
	ret.b = mat.NewDense(slipCntSum, coreCnt, nil)
	ret.db = mat.NewDense(slipCntSum, coreCnt, nil)
	ret.w = mat.NewDense(coreSizeSum, coreCnt, nil)
	ret.dw = mat.NewDense(coreSizeSum, coreCnt, nil)
	ret.ouptSize = append(slipCnt[:len(slipCnt)-1], coreCnt)
	ret.MatCaltor = NewMatCaltor(e)
	ret.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, ret.w)
	ret.Start(ret.b, ret.w, ret.db, ret.dw)
	return ret
}

func (l *HLayerConv) Predict(x *mat.Dense) (y *mat.Dense) {
	y = l.Forward(x)
	l.Clear(l.packX)
	l.packX = nil
	return
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	batch := x.RawMatrix().Cols
	wr, _ := l.w.Dims()
	br, bc := l.b.Dims()
	packX := mat.NewDense(br*batch, wr, nil)
	l.c.FoldBatches(x, packX, l.c.Pack)
	packY := mat.NewDense(br*batch, bc, nil)

	//gpu cal start
	l.packX = packX
	l.Start(l.packX, packY)
	l.Mul(packY, l.packX, l.w, false, false)
	blen := br * bc
	for j := 0; j < len(packY.RawMatrix().Data); j += blen {
		l.AddSlice(packY, l.b, j, 0, blen)
	}
	l.End(packY)
	l.Clear(packY)
	//gpu cal end

	y = mat.NewDense(br*bc, batch, nil)
	l.c.UnfoldBatches(y, packY, nil)
	return
}

func (l *HLayerConv) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	wr, _ := l.w.Dims()
	br, bc := l.b.Dims()
	l.dw.Zero()
	l.db.Zero()
	packDy := mat.NewDense(br*batch, bc, nil)
	l.c.FoldBatches(dy, packDy, nil)
	packDx := mat.NewDense(br*batch, wr, nil)

	//gpu cal start
	l.Start(packDx, packDy)
	l.Mul(l.dw, l.packX, packDy, true, false)
	blen := br * bc
	for j := 0; j < len(packDy.RawMatrix().Data); j += blen {
		l.AddSlice(l.db, packDy, 0, j, blen)
	}
	l.Mul(packDx, packDy, l.w, false, true)
	l.End(packDx)
	l.Clear(packDx, packDy, l.packX)
	//gpu cal end

	_, orgSizeSum := l.c.OrgSize()
	dx = mat.NewDense(orgSizeSum, batch, nil)
	l.c.UnfoldBatches(dx, packDx, l.c.UnPack)
	return
}

func (l *HLayerConv) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.w, l.b,
	}
	deltas = []mat.Matrix{
		l.dw, l.db,
	}
	return
}
