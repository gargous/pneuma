package rnn

import (
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

type HLayerRNN struct {
	outLay  *nn.HLayerLinear
	u       *mat.Dense
	w       *mat.Dense
	b       *mat.VecDense
	s       []*mat.Dense
	du      []*mat.Dense
	dw      []*mat.Dense
	db      []*mat.VecDense
	act     common.IHLayer
	seqSize int
	seqLen  int
	seqIdx  int
	attLen  int
}

func NewHLayerCommonRNN(seqSize, seqLen int) *HLayerRNN {
	return NewHLayerRNN(seqSize, seqLen, seqLen, nn.NewHLayerTanh())
}

func NewHLayerRNN(seqSize, seqLen, attLen int, act common.IHLayer) *HLayerRNN {
	return &HLayerRNN{
		outLay:  nn.NewHLayerLinear(),
		act:     act,
		seqSize: seqSize,
		seqLen:  seqLen,
		attLen:  attLen,
	}
}

func (l *HLayerRNN) InitSize(size []int) []int {
	r, c := size[0], size[1]
	l.w = mat.NewDense(l.seqSize, l.seqSize, nil)
	l.u = mat.NewDense(l.seqSize, r, nil)
	l.b = mat.NewVecDense(l.seqSize, nil)
	l.outLay.InitSize([]int{c, l.seqSize})
	return size
}

func (l *HLayerRNN) Forward(x *mat.Dense) (y *mat.Dense) {
	return
}

func (l *HLayerRNN) Backward(dy *mat.Dense) (dx *mat.Dense) {
	return
}

func (l *HLayerRNN) Optimize() (datas, deltas []mat.Matrix) {
	l.s = nil
	datas = []mat.Matrix{
		l.w, l.u, l.b,
	}
	deltas = []mat.Matrix{
		SumDense(l.dw), SumDense(l.du), SumVecDense(l.db),
	}
	return
}

type HLayerLSTM struct {
}
