package cnn

import (
	"math/rand"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type iHLayerOupt interface {
	OuptSize() []int
}

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	c        *ConvPacker
	w        *mat.Dense
	b        *mat.Dense
	dw       *mat.Dense
	db       *mat.Dense
	ouptSize []int
	caltor   common.IConvCaltor
}

func NewHLayerConv(inputSize, core, stride []int, padding bool) *HLayerConv {
	coreCnt := core[len(core)-1]
	ret := &HLayerConv{}
	ret.c = NewConvPacker(
		inputSize,
		core[:len(core)-1],
		stride,
		padding,
	)
	ret.b = mat.NewDense(ret.c.slipCntSum, coreCnt, nil)
	ret.w = mat.NewDense(ret.c.coreSizeSum, coreCnt, nil)
	ret.ouptSize = append(ret.c.slipCnt[:len(ret.c.slipCnt)-1], coreCnt)
	ret.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, ret.w)
	ret.caltor = &ConvCalter{}
	return ret
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	wr, _ := l.w.Dims()
	br, bc := l.b.Dims()
	batch := x.RawMatrix().Cols
	packX := mat.NewDense(br*batch, wr, nil)
	for j := 0; j < batch; j++ {
		xCol := x.ColView(j).(*mat.VecDense)
		sliceX := packX.Slice(j*br, j*br+br, 0, wr).(*mat.Dense)
		packXCol := l.c.Pack(xCol)
		sliceX.Copy(packXCol)
	}
	packY := mat.NewDense(br*batch, bc, nil)
	l.caltor.Forward(packX, packY, l.w, l.b)
	y = mat.NewDense(br*bc, batch, nil)
	for j := 0; j < batch; j++ {
		sliceY := packY.Slice(j*br, j*br+br, 0, bc).(*mat.Dense)
		y.SetCol(j, sliceY.RawMatrix().Data)
	}
	return
}

func (l *HLayerConv) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	wr, wc := l.w.Dims()
	br, bc := l.b.Dims()
	l.dw = mat.NewDense(wr, wc, nil)
	l.db = mat.NewDense(br, bc, nil)
	packDy := mat.NewDense(br*batch, bc, dy.RawMatrix().Data)
	packDx := mat.NewDense(br*batch, wr, nil)
	l.caltor.Backward(packDx, packDy, l.dw, l.db)
	dx = mat.NewDense(l.c.orgSizeSum, batch, nil)
	for j := 0; j < batch; j++ {
		sliceDx := packDx.Slice(j*br, j*br+br, 0, wr).(*mat.Dense)
		dx.SetCol(j, l.c.UnPack(sliceDx).RawVector().Data)
	}
	return
}

func (l *HLayerConv) SetCaltor(caltor common.IConvCaltor) {
	l.caltor = caltor
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

func (l *HLayerConv) OuptSize() []int {
	return l.ouptSize
}

type HLayerDimBatchNorm struct {
	picker *MatColPicker
	batch  *nn.HLayerBatchNorm
}

func NewHLayerConvBatchNorm(inputSize []int, minstd, momentum float64) *HLayerDimBatchNorm {
	return &HLayerDimBatchNorm{
		picker: NewMatColPicker(inputSize, len(inputSize)-1),
		batch:  nn.NewHLayerBatchNorm(minstd, momentum),
	}
}

func (l *HLayerDimBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	newX := l.picker.Pick(x)
	newXT := mat.DenseCopyOf(newX.T())
	newYT := l.batch.Forward(newXT)
	newY := mat.DenseCopyOf(newYT.T())
	return l.picker.Pick(newY)
}

func (l *HLayerDimBatchNorm) Backward(dy *mat.Dense) (dx *mat.Dense) {
	newDy := l.picker.Pick(dy)
	newDyT := mat.DenseCopyOf(newDy.T())
	newDxT := l.batch.Backward(newDyT)
	newDx := mat.DenseCopyOf(newDxT.T())
	return l.picker.Pick(newDx)
}
func (l *HLayerDimBatchNorm) Optimize() (datas, deltas []mat.Matrix) {
	return l.batch.Optimize()
}

type HLayerMaxPooling struct {
	c        *ConvPacker
	inptSize []int
	ouptSize []int
	slips    []*mat.Dense
}

func NewHLayerMaxPooling(inputSize, core, stride []int, padding bool) *HLayerMaxPooling {
	ret := &HLayerMaxPooling{}
	inptCnt := inputSize[len(inputSize)-1]
	ret.c = NewConvPacker(
		inputSize,
		core,
		stride,
		padding,
	)
	ret.ouptSize = append(ret.c.slipCnt[:len(ret.c.slipCnt)-1], inptCnt)
	ret.inptSize = inputSize
	return ret
}

func (l *HLayerMaxPooling) Forward(x *mat.Dense) (y *mat.Dense) {
	batch := x.RawMatrix().Cols
	cnt := l.inptSize[len(l.inptSize)-1]
	y = mat.NewDense(l.c.slipCntSum*cnt, batch, nil)
	l.slips = make([]*mat.Dense, batch)
	for j := 0; j < batch; j++ {
		xCol := x.ColView(j).(*mat.VecDense)
		yCol := y.ColView(j).(*mat.VecDense)
		slip := l.c.Pack(xCol)
		newSlip := mat.NewDense(l.c.slipCntSum, l.c.coreSizeSum, nil)
		recuRange([]int{l.c.slipCntSum, cnt}, nil, func(pos []int) {
			si, sj := pos[0], pos[1]
			slipRow := slip.RowView(si).(*mat.VecDense)
			newSlipRow := newSlip.RowView(si).(*mat.VecDense)
			slipSlice := make([]float64, l.c.coreSizeSum/cnt)
			for sk := 0; sk < len(slipSlice); sk++ {
				xidx := idxExpend(sk, sj, cnt)
				slipSlice[sk] = slipRow.AtVec(xidx)
			}
			maxIdx := floats.MaxIdx(slipSlice)
			yCol.SetVec(idxExpend(si, sj, cnt), slipSlice[maxIdx])
			newSlipRow.SetVec(idxExpend(maxIdx, sj, cnt), 1)
		})
		l.slips[j] = newSlip
	}
	return
}

func (l *HLayerMaxPooling) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	cnt := l.inptSize[len(l.inptSize)-1]
	dx = mat.NewDense(l.c.orgSizeSum, batch, nil)
	for j := 0; j < batch; j++ {
		dxCol := dx.ColView(j).(*mat.VecDense)
		slip := l.slips[j]
		dyColMat := mat.NewDense(l.c.slipCntSum, cnt, nil)
		for k := 0; k < l.c.coreSizeSum/cnt; k++ {
			slipSlice := slip.Slice(0, l.c.slipCntSum, k*cnt, k*cnt+cnt).(*mat.Dense)
			slipSlice.MulElem(slipSlice, dyColMat)
		}
		dxCol.CopyVec(l.c.UnPack(slip))
		dy.SetCol(j, dyColMat.RawMatrix().Data)
	}
	return
}

func (l *HLayerMaxPooling) OuptSize() []int {
	return l.ouptSize
}
