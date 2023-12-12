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
	coreCnt := core[0]
	inptCnt := inputSize[0]
	dim := len(inputSize)
	ret := &HLayerConv{
		inptSize:     inputSize,
		coreSize:     append([]int{inptCnt}, core[1:]...),
		stride:       append([]int{inptCnt}, stride...),
		paddingLeft:  make([]int, dim),
		paddingRight: make([]int, dim),
		fitSize:      make([]int, dim),
		ouptSize:     make([]int, dim),
	}
	slips := make([]int, dim)
	for i := 0; i < dim; i++ {
		iinp := inputSize[i]
		icor := ret.coreSize[i]
		istr := ret.stride[i]
		pl, pr, slip := paddingCnt(iinp, icor, istr, padding)
		ret.paddingLeft[i], ret.paddingRight[i] = pl, pr
		slips[i] = slip
	}
	ret.b = mat.NewDense(intsProd(slips), coreCnt, nil)
	ret.w = mat.NewDense(intsProd(ret.coreSize), coreCnt, nil)
	ret.ouptSize = append([]int{coreCnt}, slips[1:]...)
	for i := 0; i < dim; i++ {
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

func (l *HLayerConv) slipBuild(data *mat.VecDense) *mat.Dense {
	orgSize := l.inptSize
	fitSize := l.fitSize
	step := len(orgSize) - 1
	fitData := data
	if !intsEqual(orgSize, fitSize) {
		fitData = mat.NewVecDense(intsProd(fitSize), nil)
		padStep, pad := l.paddingLeft[:step], l.paddingLeft[step]
		offset := intsMin([]int{orgSize[step], fitSize[step]})
		orgStepSize := orgSize[:step]
		fitStepSize := fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			fitStepPos := intsAdd(orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgIdx := posIdx(append(orgStepPos, 0), orgSize)
			fitIdx := posIdx(append(fitStepPos, 0), fitSize)
			if pad < 0 {
				orgIdx -= pad
			} else {
				fitIdx += pad
			}
			fitSlice := fitData.SliceVec(fitIdx, fitIdx+offset).(*mat.VecDense)
			orgSlice := data.SliceVec(orgIdx, orgIdx+offset)
			fitSlice.CopyVec(orgSlice)
		})
	}
	slipRowLen, _ := l.w.Dims()
	slipColLen, _ := l.b.Dims()
	slipMat := mat.NewDense(slipColLen, slipRowLen, nil)
	slipRowIdx := 0
	recuRange(fitSize, l.stride, func(startPos []int) {
		slipRow := slipMat.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		coreStepSize := l.coreSize[:step]
		coreStepOffset := l.coreSize[step]
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			fitStepPos := intsAdd(coreStepPos, startStepPos)
			fitIdx := posIdx(append(fitStepPos, startStepOffset), fitSize)
			slipIdx := posIdx(append(coreStepPos, 0), l.coreSize)
			fitRow := fitData.SliceVec(fitIdx, fitIdx+coreStepOffset)
			slipRow.SliceVec(slipIdx, slipIdx+coreStepOffset).(*mat.VecDense).CopyVec(fitRow)
		})
	})
	return slipMat
}

func (l *HLayerConv) slipRestore(data *mat.Dense) *mat.VecDense {
	orgSize := l.inptSize
	fitSize := l.fitSize
	step := len(orgSize) - 1
	slipRowIdx := 0
	fitData := mat.NewVecDense(intsProd(fitSize), nil)
	recuRange(fitSize, l.stride, func(startPos []int) {
		slipRow := data.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		coreStepSize := l.coreSize[:step]
		coreStepOffset := l.coreSize[step]
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			fitStepPos := intsAdd(coreStepPos, startStepPos)
			fitIdx := posIdx(append(fitStepPos, startStepOffset), fitSize)
			slipIdx := posIdx(append(coreStepPos, 0), l.coreSize)
			fitRow := fitData.SliceVec(fitIdx, fitIdx+coreStepOffset).(*mat.VecDense)
			fitRow.AddVec(fitRow, slipRow.SliceVec(slipIdx, slipIdx+coreStepOffset))
		})
	})
	retData := fitData
	if !intsEqual(orgSize, fitSize) {
		retData = mat.NewVecDense(intsProd(orgSize), nil)
		padStep, pad := l.paddingLeft[:step], l.paddingLeft[step]
		offset := intsMin([]int{orgSize[step], fitSize[step]})
		orgStepSize := orgSize[:step]
		fitStepSize := fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			fitStepPos := intsAdd(orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgIdx := posIdx(append(orgStepPos, 0), orgSize)
			fitIdx := posIdx(append(fitStepPos, 0), fitSize)
			if pad < 0 {
				orgIdx -= pad
			} else {
				fitIdx += pad
			}
			retSlice := retData.SliceVec(orgIdx, orgIdx+offset).(*mat.VecDense)
			fitSlice := fitData.SliceVec(fitIdx, fitIdx+offset)
			retSlice.CopyVec(fitSlice)
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
		xColData := x.ColView(j).(*mat.VecDense)
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
		dx.SetCol(j, l.slipRestore(slipDx).RawVector().Data)
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

type HLayerAvePooling struct {
	inptSize []int
	ouptSize []int
	coreSize []int
}

func NewHLayerAvePooling(inputSize, core []int) *HLayerAvePooling {
	ret := &HLayerAvePooling{}
	ret.inptSize = inputSize
	ret.coreSize = append([]int{1}, core...)
	ret.ouptSize = make([]int, len(inputSize))
	ret.ouptSize[0] = inputSize[0]
	for i := 0; i < len(inputSize)-1; i++ {
		_, _, cnt := paddingCnt(inputSize[i+1], core[i], core[i], true)
		ret.ouptSize[i+1] = cnt
	}
	return ret
}

func (l *HLayerAvePooling) Forward(x *mat.Dense) (y *mat.Dense) {
	_, batch := x.Dims()
	y = mat.NewDense(intsProd(l.ouptSize), batch, nil)
	step := len(l.inptSize) - 1
	for j := 0; j < batch; j++ {
		onex := x.ColView(j).(*mat.VecDense)
		oney := y.ColView(j).(*mat.VecDense)
		recuRange(l.inptSize, l.coreSize, func(startPos []int) {
			startPosFL := []int{startPos[0]}
			startStepPos := startPos[1:step]
			startStepOffset := startPos[step]
			coreStepSize := l.coreSize[1:step]
			coreStepOffset := l.coreSize[step]
			stepBound := l.inptSize[step]
			sum := 0.0
			recuRange(coreStepSize, nil, func(coreStepPos []int) {
				stepPos := intsAdd(coreStepPos, startStepPos)
				if !sizeBound(stepPos, l.inptSize[1:step]) {
					return
				}
				idx := posIdx(append(append(startPosFL, stepPos...), startStepOffset), l.inptSize)
				idxEnd := idx + coreStepOffset
				if idxEnd > stepBound {
					idxEnd = stepBound
				}
				fitRow := onex.SliceVec(idx, idxEnd)
				sum += mat.Sum(fitRow)
			})
			cnt := intsProd(l.coreSize[1:])
			oney.SetVec(posIdx(startPos, l.ouptSize), sum/float64(cnt))
		})
	}
	return
}

func (l *HLayerAvePooling) Backward(dy *mat.Dense) (dx *mat.Dense) {
	return
}

func (l *HLayerAvePooling) OuptSize() []int {
	return l.ouptSize
}
