package frcnn

import (
	"pneuma/cnn"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type ModelBuilder struct {
	roiSize []int
	c       *cnn.ModelSizeBuilder
	*nn.ModelBuilder
	model *Model
	rpn   func(score, trans cnn.ConvKernalParam) (scnv, tcnv common.IHLayerSizeIniter, opt common.IOptimizer)
}

func NewModelBuilder(size, roiSize []int) *ModelBuilder {
	return &ModelBuilder{
		roiSize: roiSize,
		c:       cnn.NewModelSizeBuilder(size),
		model:   NewModel(),
	}
}

func (b *ModelBuilder) C(cb func(*nn.ModelSample)) {
	b.c.One().Use(cb)
}

func (b *ModelBuilder) Cs(cnt int, cb func(i int, ms *nn.ModelSample)) {
	for i := 0; i < cnt; i++ {
		b.c.One().Use(func(ms *nn.ModelSample) {
			cb(i, ms)
		})
	}
}

func (b *ModelBuilder) CLay(l func() common.IHLayer) {
	b.c.Lay(l)
}
func (b *ModelBuilder) COpt(l func() common.IOptimizer) {
	b.c.Opt(l)
}

func (b *ModelBuilder) RPN(l func(score, trans cnn.ConvKernalParam) (scnv, tcnv common.IHLayerSizeIniter, opt common.IOptimizer)) {
	b.rpn = l
}

func (b *ModelBuilder) Build() *Model {
	m := b.model
	csize := b.c.Bulld(m.Model)
	if b.rpn != nil {
		param := NewRPNParam(b.c.Size, b.roiSize)
		m.UseRPN(param)
		sconv, tconv, opt := b.rpn(m.RPN.ScoreLayerParam(), m.RPN.TransLayerParam())
		m.RPN.SetScoreLayer(sconv)
		m.RPN.SetTransLayer(tconv)
		m.RPN.SetOpt(opt)
		m.RPN.InitSize(csize)
	}
	return m
}

type Model struct {
	*nn.Model
	RPN *RPN
}

func NewModel() *Model {
	ret := &Model{
		Model: nn.NewModel(),
	}
	return ret
}

func (m *Model) UseRPN(param *RPNParam) *RPN {
	m.RPN = NewRPN(param)
	m.RPN.SetTrsTarget(nn.NewTarSmoothMAE(0.5), NewRPNLossParam())
	return m.RPN
}

func (m *Model) Test(x, bnds *mat.Dense) (loss, acc float64) {
	a := m.Model.Predict(x)
	if m.RPN != nil {
		return m.RPN.Test(a, bnds)
	}
	return
}

func (m *Model) Train(x, bnds *mat.Dense) *mat.Dense {
	a := m.Model.Forward(x)
	var da *mat.Dense
	if m.RPN != nil {
		da = m.RPN.Train(a, bnds)
	}
	if da == nil {
		return nil
	}
	da = m.Model.Backward(da)
	m.Model.Update()
	return da
}

func (m *Model) PredicTG(x *mat.Dense) *mat.Dense {
	a := m.Model.Predict(x)
	if m.RPN != nil {
		return m.RPN.PredicTG(a)
	}
	return nil
}

func (m *Model) Tests(x, y []*mat.Dense) (loss, acc float64) {
	cnt := len(x)
	for i := 0; i < cnt; i++ {
		oneLoss, oneAcc := m.Test(x[i], y[i])
		loss += oneLoss
		acc += oneAcc
	}
	return loss / float64(cnt), acc / float64(cnt)
}

func (m *Model) Trains(trainX, trainY []*mat.Dense) {
	for i := range trainX {
		x, y := trainX[i], trainY[i]
		m.Train(x, y)
		if m.IsDone() {
			return
		}
	}
}

func (m *Model) TrainTimes(trainX, trainY []*mat.Dense, oneTimes func(int, int)) {
	nn.WithTimes(len(trainX), func(i int) bool {
		x, y := trainX[i], trainY[i]
		m.Train(x, y)
		return !m.IsDone()
	}, oneTimes)
}

func (m *Model) IsDone() bool {
	if m.RPN != nil {
		return m.RPN.loss.isDone()
	}
	return true
}

func (m *Model) LossPopMean() float64 {
	if m.RPN != nil {
		mean := stat.Mean(m.RPN.loss.losses, nil)
		m.RPN.loss.losses = nil
		return mean
	}
	return 0
}

func (m *Model) LossLatest() float64 {
	if m.RPN != nil {
		loss := m.RPN.loss.losses
		if len(loss) > 0 {
			return loss[len(loss)-1]
		}
	}
	return 0
}
