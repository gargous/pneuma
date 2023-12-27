package cnn

import (
	"math/rand"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

// w b均为展开状态，每列一个卷积核
type HLayerConv struct {
	C     *ConvPacker
	W     *mat.Dense
	B     *mat.Dense
	DW    *mat.Dense
	DB    *mat.Dense
	PackX *mat.Dense
	param ConvKernalParam
}

func NewHLayerConv(param ConvKernalParam) *HLayerConv {
	return &HLayerConv{param: param}
}

func (l *HLayerConv) InitSize(size []int) []int {
	coreCnt := l.param.size[len(l.param.size)-1]
	l.param.size = l.param.size[:len(l.param.size)-1]
	l.C = NewConvPacker(size, l.param)
	l.B = mat.NewDense(l.C.slipCntSum, coreCnt, nil)
	l.DB = mat.NewDense(l.C.slipCntSum, coreCnt, nil)
	l.W = mat.NewDense(l.C.coreSizeSum, coreCnt, nil)
	l.W.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, l.W)
	l.DW = mat.NewDense(l.C.coreSizeSum, coreCnt, nil)
	return append(l.C.slipCnt[:len(l.C.slipCnt)-1], coreCnt)
}

func (l *HLayerConv) Forward(x *mat.Dense) (y *mat.Dense) {
	batch := x.RawMatrix().Cols
	wr, _ := l.W.Dims()
	br, bc := l.B.Dims()
	packX := mat.NewDense(br*batch, wr, nil)
	l.C.FoldBatches(x, packX, l.C.PackTo)
	packY := mat.NewDense(br*batch, bc, nil)
	l.PackX = packX

	packY.Mul(packX, l.W)
	blen := br * bc
	packYData := packY.RawMatrix().Data
	for j := 0; j < len(packYData); j += blen {
		sliceY := mat.NewDense(br, bc, packYData[j:j+blen])
		sliceY.Add(sliceY, l.B)
	}

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

	l.DW.Mul(l.PackX.T(), packDy)
	blen := br * bc
	packDyData := packDy.RawMatrix().Data
	for j := 0; j < len(packDyData); j += blen {
		sliceDy := mat.NewDense(br, bc, packDyData[j:j+blen])
		l.DB.Add(l.DB, sliceDy)
	}
	packDx.Mul(packDy, l.W.T())

	dx = mat.NewDense(l.C.orgSizeSum, batch, nil)
	l.C.UnfoldBatches(dx, packDx, l.C.UnPackTo)
	return
}

func (l *HLayerConv) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.W, l.B,
	}
	deltas = []mat.Matrix{
		l.DW, l.DB,
	}
	return
}

type HLayerDimBatchNorm struct {
	*nn.HLayerBatchNorm
	picker   *MatColPicker
	batchDim int
}

func NewHLayerConvBatchNorm(minstd, momentum float64) *HLayerDimBatchNorm {
	return &HLayerDimBatchNorm{
		batchDim:        -1,
		HLayerBatchNorm: nn.NewHLayerBatchNorm(minstd, momentum),
	}
}

func (l *HLayerDimBatchNorm) InitSize(size []int) []int {
	batchDim := (l.batchDim + len(size)) % len(size)
	l.picker = NewMatColPicker(size, batchDim)
	l.HLayerBatchNorm.InitSize([]int{size[batchDim]})
	return size
}

func (l *HLayerDimBatchNorm) Predict(x *mat.Dense) (y *mat.Dense) {
	newX := l.picker.Pick(x)
	newXT := mat.DenseCopyOf(newX.T())
	newYT := l.HLayerBatchNorm.Predict(newXT)
	newY := mat.DenseCopyOf(newYT.T())
	return l.picker.Pick(newY)
}

func (l *HLayerDimBatchNorm) Forward(x *mat.Dense) (y *mat.Dense) {
	newX := l.picker.Pick(x)
	newXT := mat.DenseCopyOf(newX.T())
	newYT := l.HLayerBatchNorm.Forward(newXT)
	newY := mat.DenseCopyOf(newYT.T())
	return l.picker.Pick(newY)
}

func (l *HLayerDimBatchNorm) Backward(dy *mat.Dense) (dx *mat.Dense) {
	newDy := l.picker.Pick(dy)
	newDyT := mat.DenseCopyOf(newDy.T())
	newDxT := l.HLayerBatchNorm.Backward(newDyT)
	newDx := mat.DenseCopyOf(newDxT.T())
	return l.picker.Pick(newDx)
}

type MaxPoolingCalInfo struct {
	search []int
	idxes  []int
	cnt    int
}

type HLayerMaxPooling struct {
	C        *ConvPacker
	PackX    *mat.Dense
	inptSize []int
	param    ConvKernalParam
	info     MaxPoolingCalInfo
}

func NewHLayerMaxPooling(param ConvKernalParam) *HLayerMaxPooling {
	return &HLayerMaxPooling{param: param}
}

func (l *HLayerMaxPooling) InitSize(size []int) []int {
	inptCnt := size[len(size)-1]
	l.C = NewConvPacker(size, l.param)
	l.inptSize = size
	span := l.C.coreSizeSum / inptCnt
	l.info = MaxPoolingCalInfo{
		search: []int{span, inptCnt},
		idxes:  make([]int, inptCnt),
		cnt:    inptCnt,
	}
	return append(l.C.slipCnt[:len(l.C.slipCnt)-1], inptCnt)
}

func (l *HLayerMaxPooling) Forward(x *mat.Dense) (y *mat.Dense) {
	batch := x.RawMatrix().Cols
	packX := mat.NewDense(l.C.slipCntSum*batch, l.C.coreSizeSum, nil)
	packY := mat.NewDense(l.C.slipCntSum*batch, l.info.cnt, nil)
	l.C.FoldBatches(x, packX, l.C.PackTo)
	l.PackX = packX
	for i := 0; i < l.C.slipCntSum*batch; i++ {
		rowX := packX.RawRowView(i)
		rowY := packY.RawRowView(i)
		common.IntsSetConst(-1, l.info.idxes)
		common.RecuRange(l.info.search, nil, func(pos []int) {
			k := pos[len(pos)-1]
			idx := common.PosIdx(pos, l.info.search)
			if rowX[idx] > rowY[k] || l.info.idxes[k] < 0 {
				l.info.idxes[k] = idx
				rowY[k] = rowX[idx]
			}
			rowX[idx] = 0
		})
		for k := 0; k < l.info.cnt; k++ {
			rowX[l.info.idxes[k]] = 1
		}
	}
	y = mat.NewDense(l.C.slipCntSum*l.info.cnt, batch, nil)
	l.C.UnfoldBatches(y, packY, nil)
	return
}

func (l *HLayerMaxPooling) Backward(dy *mat.Dense) (dx *mat.Dense) {
	batch := dy.RawMatrix().Cols
	packDy := mat.NewDense(l.C.slipCntSum*batch, l.info.cnt, nil)
	packDx := l.PackX
	l.C.FoldBatches(dy, packDy, nil)
	for j := 0; j < l.C.coreSizeSum; j++ {
		colDy := packDy.ColView(j % l.info.cnt)
		colDx := packDx.ColView(j).(*mat.VecDense)
		colDx.MulElemVec(colDx, colDy)
	}
	dx = mat.NewDense(l.C.orgSizeSum, batch, nil)
	l.C.UnfoldBatches(dx, packDx, l.C.UnPackTo)
	return
}
