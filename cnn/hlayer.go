package cnn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	inptSize     []int
	ouptSize     []int
	fitSize      []int
	coreSize     []int
	stride       []int
	paddingLeft  []int
	paddingRight []int
	w            *mat.Dense
	b            *mat.Dense
	dw           *mat.Dense
	db           *mat.Dense
	slips        []*mat.Dense
}

func NewHLayerConv(inputSize, core, stride []int, padding bool) *HLayerConv {
	dim := len(inputSize)
	ret := &HLayerConv{
		inptSize:     inputSize,
		coreSize:     core[:dim-1],
		stride:       stride[:dim-1],
		paddingLeft:  make([]int, dim-1),
		paddingRight: make([]int, dim-1),
		fitSize:      make([]int, dim-1),
		ouptSize:     make([]int, dim),
	}
	slips := make([]int, dim-1)
	for i := 0; i < dim-1; i++ {
		iinp := inputSize[i]
		icor := core[i]
		istr := stride[i]
		pl, pr, slip := paddingCnt(iinp, icor, istr, padding)
		ret.paddingLeft[i], ret.paddingRight[i] = pl, pr
		slips[i] = slip
	}
	coreCnt := core[dim-1]
	ret.b = mat.NewDense(intsProd(slips), coreCnt, nil)
	ret.w = mat.NewDense(intsProd(append(ret.coreSize, inputSize[dim-1])), coreCnt, nil)
	ret.ouptSize = append(slips, coreCnt)
	for i := 0; i < dim-1; i++ {
		ret.fitSize[i] = (ret.paddingLeft[i] + inputSize[i] + ret.paddingRight[i])
	}
	ret.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, ret.w)
	ret.b.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, ret.b)
	return ret
}

func (l *HLayerConv) slipBuild(data []float64) *mat.Dense {
	orgSize := l.inptSize[:len(l.inptSize)-1]
	fitSize := l.fitSize
	chLen := l.inptSize[len(l.inptSize)-1]
	fitData := data
	if !intsEqual(orgSize, fitSize) {
		fitData = make([]float64, intsProd(fitSize)*chLen)
		recuRange(orgSize, nil, func(orgPos []int) {
			fitPos := intsAdd(orgPos, l.paddingLeft)
			if !sizeBound(fitPos, fitSize) {
				return
			}
			orgIdx := idxExpend(posIdx(orgPos, orgSize), 0, chLen)
			fitIdx := idxExpend(posIdx(fitPos, fitSize), 0, chLen)
			copy(fitData[fitIdx:fitIdx+chLen], data[orgIdx:orgIdx+chLen])
		})
	}
	slipRowLen, _ := l.w.Dims()
	slipColLen, _ := l.b.Dims()
	slipMat := mat.NewDense(slipColLen, slipRowLen, nil)
	slipRowIdx := 0
	recuRange(fitSize, l.stride, func(startPos []int) {
		slipRow := slipMat.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		recuRange(l.coreSize, nil, func(pos []int) {
			fitPos := intsAdd(pos, startPos)
			fitIdx := idxExpend(posIdx(fitPos, fitSize), 0, chLen)
			slipIdx := idxExpend(posIdx(pos, l.coreSize), 0, chLen)
			fixRow := mat.NewVecDense(chLen, fitData[fitIdx:fitIdx+chLen])
			slipRow.SliceVec(slipIdx, slipIdx+chLen).(*mat.VecDense).CopyVec(fixRow)
		})
	})
	return slipMat
}

func (l *HLayerConv) slipRestore(data *mat.Dense) []float64 {
	orgSize := l.inptSize[:len(l.inptSize)-1]
	fitSize := l.fitSize
	chLen := l.inptSize[len(l.inptSize)-1]
	slipRowIdx := 0
	fitData := make([]float64, intsProd(fitSize)*chLen)
	recuRange(fitSize, l.stride, func(startPos []int) {
		slipRow := data.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		recuRange(l.coreSize, nil, func(pos []int) {
			fitPos := intsAdd(pos, startPos)
			fitIdx := idxExpend(posIdx(fitPos, fitSize), 0, chLen)
			slipIdx := idxExpend(posIdx(pos, l.coreSize), 0, chLen)
			fixRow := mat.NewVecDense(chLen, fitData[fitIdx:fitIdx+chLen])
			fixRow.AddVec(fixRow, slipRow.SliceVec(slipIdx, slipIdx+chLen))
		})
	})
	retData := fitData
	if !intsEqual(orgSize, fitSize) {
		retData = make([]float64, intsProd(orgSize)*chLen)
		recuRange(orgSize, nil, func(orgPos []int) {
			fixPos := intsAdd(orgPos, l.paddingLeft)
			if !sizeBound(fixPos, fitSize) {
				return
			}
			orgIdx := idxExpend(posIdx(orgPos, orgSize), 0, chLen)
			fitIdx := idxExpend(posIdx(fixPos, fitSize), 0, chLen)
			copy(retData[orgIdx:orgIdx+chLen], fitData[fitIdx:fitIdx+chLen])
		})
	}
	return retData
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	slipCnt, coreCnt := l.b.Dims()
	batch := x.RawMatrix().Cols
	y = mat.NewDense(slipCnt*coreCnt, batch, nil)
	l.slips = make([]*mat.Dense, batch)
	for j := 0; j < batch; j++ {
		xColData := x.ColView(j).(*mat.VecDense).RawVector().Data
		yColData := y.ColView(j).(*mat.VecDense).RawVector().Data
		slip := l.slipBuild(xColData)
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
	dx = mat.NewDense(intsProd(l.inptSize), batch, nil)
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
		dx.SetCol(j, l.slipRestore(slipDx))
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
