package cnn

import (
	"math/rand"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	c        *ConvInfo
	w        *mat.Dense
	b        *mat.Dense
	dw       *mat.Dense
	db       *mat.Dense
	ouptSize []int
	slips    []*mat.Dense
}

func NewHLayerConv(inputSize, core, stride []int, padding bool) *HLayerConv {
	coreCnt := core[len(core)-1]
	ret := &HLayerConv{}
	ret.c = NewConvInfo(
		inputSize,
		core[:len(core)-1],
		stride,
		padding,
	)
	ret.b = mat.NewDense(ret.c.slipCntSum, coreCnt, nil)
	ret.w = mat.NewDense(ret.c.coreSizeSum, coreCnt, nil)
	ret.ouptSize = append(ret.c.slipCnt[:len(ret.c.slipCnt)-1], coreCnt)
	ret.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, ret.w)
	ret.b.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, ret.b)
	return ret
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	slipCnt, coreCnt := l.b.Dims()
	batch := x.RawMatrix().Cols
	y = mat.NewDense(slipCnt*coreCnt, batch, nil)
	l.slips = make([]*mat.Dense, batch)
	for j := 0; j < batch; j++ {
		xCol := x.ColView(j).(*mat.VecDense)
		yColData := y.ColView(j).(*mat.VecDense).RawVector().Data
		slip := l.c.slipBuild(xCol)
		l.slips[j] = slip
		oneRet := mat.NewDense(slipCnt, coreCnt, yColData)
		oneRet.Mul(slip, l.w)
		oneRet.Add(oneRet, l.b)
	}
	return
}

func (l *HLayerConv) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	wr, wc := l.w.Dims()
	br, bc := l.b.Dims()
	l.dw = mat.NewDense(wr, wc, nil)
	l.db = mat.NewDense(br, bc, nil)
	dx = mat.NewDense(l.c.orgSizeSum, batch, nil)
	for j := 0; j < batch; j++ {
		dyColData := dy.ColView(j).(*mat.VecDense).RawVector().Data
		slipDy := mat.NewDense(br, bc, dyColData)
		slipX := l.slips[j]
		slipDw := mat.NewDense(wr, wc, nil)
		slipDw.Mul(slipX.T(), slipDy)
		l.dw.Add(l.dw, slipDw)
		l.db.Add(l.db, slipDy)
		sxr, sxc := slipX.Dims()
		slipDx := mat.NewDense(sxr, sxc, nil)
		slipDx.Mul(slipDy, l.w.T())
		dx.SetCol(j, l.c.slipRestore(slipDx).RawVector().Data)
	}
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

func (l *HLayerConv) OuptSize() []int {
	return l.ouptSize
}

type HLayerMaxPooling struct {
	c        *ConvInfo
	inptSize []int
	ouptSize []int
	slips    []*mat.Dense
}

func NewHLayerMaxPooling(inputSize, core, stride []int, padding bool) *HLayerMaxPooling {
	ret := &HLayerMaxPooling{}
	inptCnt := inputSize[len(inputSize)-1]
	ret.c = NewConvInfo(
		inputSize,
		core,
		stride,
		padding,
	)
	ret.ouptSize = append(ret.c.slipCnt, inptCnt)
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
		slip := l.c.slipBuild(xCol)
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
		dyCol := dy.ColView(j).(*mat.VecDense)
		slip := l.slips[j]
		dyColMat := mat.NewDense(l.c.slipCntSum, cnt, dyCol.RawVector().Data)
		for k := 0; k < l.c.coreSizeSum/cnt; k++ {
			slipSlice := slip.Slice(0, l.c.slipCntSum, k*cnt, k*cnt+cnt).(*mat.Dense)
			slipSlice.MulElem(slipSlice, dyColMat)
		}
		dxCol.CopyVec(l.c.slipRestore(slip))
	}
	return
}

func (l *HLayerMaxPooling) OuptSize() []int {
	return l.ouptSize
}
