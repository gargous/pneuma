package nn

import (
	"pneuma/common"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type ModelSample struct {
	Lays      []common.IHLayer
	Optimizer common.IOptimizer
}

func (m *ModelSample) Lay(l common.IHLayer) {
	m.Lays = append(m.Lays, l)
}

func (m *ModelSample) Opt(l common.IOptimizer) {
	m.Optimizer = l
}
func (m *ModelSample) Use(cb func(*ModelSample)) {
	cb(m)
}

type ModelBuilder struct {
	Lays      []func() common.IHLayer
	Optimizer func() common.IOptimizer
}

func (m *ModelBuilder) Lay(l func() common.IHLayer) {
	m.Lays = append(m.Lays, l)
}

func (m *ModelBuilder) Opt(l func() common.IOptimizer) {
	m.Optimizer = l
}

type Model struct {
	layers []*layer
	loss   *loss
}

func NewModel() *Model {
	return &Model{}
}

func (m *Model) SetTarget(tar common.ITarget, param *LossParam) {
	m.SetLoss(tar, param, nil)
}

func (m *Model) SetLoss(tar common.ITarget, param *LossParam, lossVal []float64) {
	m.loss = &loss{
		target: tar,
		param:  param,
		losses: lossVal,
	}
}

func (m *Model) Target() (tar common.ITarget, param *LossParam) {
	tar = m.loss.target
	param = m.loss.param
	return
}

func (m *Model) AddLayer(opt common.IOptimizer, layers ...common.IHLayer) {
	m.layers = append(m.layers, &layer{
		optimizer: opt,
		hlayers:   layers,
	})
}

func (m *Model) LayerCnt() int {
	return len(m.layers)
}

func (m *Model) Layer(idx int) (opt common.IOptimizer, layers []common.IHLayer) {
	layer := m.layers[idx]
	opt = layer.optimizer
	layers = layer.hlayers
	return
}

func (m *Model) SetLayer(idx int, opt common.IOptimizer, layers ...common.IHLayer) {
	if optc, isOptc := opt.(common.IOptimizerCoLayer); isOptc {
		var clayers []common.IHLayerOptimizer
		for i := 0; i < len(layers); i++ {
			if layc, isLayc := layers[i].(common.IHLayerOptimizer); isLayc {
				clayers = append(clayers, layc)
			}
		}
		optc.SetIHLayers(clayers...)
	}
	m.layers[idx] = &layer{
		optimizer: opt,
		hlayers:   layers,
	}
}

func (m *Model) Forward(x *mat.Dense) *mat.Dense {
	for i := 0; i < len(m.layers); i++ {
		x = m.layers[i].forward(x)
	}
	return x
}

func (m *Model) Backward(dx *mat.Dense) *mat.Dense {
	for i := len(m.layers) - 1; i >= 0; i-- {
		dx = m.layers[i].backward(dx)
	}
	return dx
}

func (m *Model) Update() {
	for i := 0; i < len(m.layers); i++ {
		m.layers[i].update()
	}
}

func (m *Model) Train(x, y *mat.Dense) *mat.Dense {
	a := m.Forward(x)
	m.loss.forward(a, y)
	if m.loss.isDone() {
		return nil
	}
	da := m.Backward(m.loss.backward())
	m.Update()
	return da
}

func (m *Model) Predict(x *mat.Dense) *mat.Dense {
	a := x
	for i := 0; i < len(m.layers); i++ {
		a = m.layers[i].predict(a)
	}
	return a
}

func (m *Model) Test(x, y *mat.Dense) (loss, acc float64) {
	pred := m.Predict(x)
	loss = m.loss.target.Loss(pred, y)
	acc = m.loss.target.Acc(pred, y)
	return
}

func (m *Model) Tests(x, y []*mat.Dense) (loss, acc float64) {
	cnt := len(x)
	for i := 0; i < cnt; i++ {
		oneloss, oneacc := m.Test(x[i], y[i])
		loss += oneloss
		acc += oneacc
	}
	return loss / float64(cnt), acc / float64(cnt)
}

func (m *Model) LossPopMean() float64 {
	mean := stat.Mean(m.loss.losses, nil)
	m.loss.losses = nil
	return mean
}

func (m *Model) LossLatest() float64 {
	if len(m.loss.losses) <= 0 {
		return 0
	}
	return m.loss.losses[len(m.loss.losses)-1]
}

func (m *Model) IsDone() bool {
	return m.loss.isDone()
}

func (m *Model) Predicts(predX []*mat.Dense) []*mat.Dense {
	predY := make([]*mat.Dense, len(predX))
	for i := 0; i < len(predX); i++ {
		predY[i] = m.Predict(predX[i])
	}
	return predY
}

func WithTimes(cnt int, do func(int) bool, oneTimes func(int, int)) {
	for i := 0; i < cnt; i++ {
		var stTime time.Time
		if oneTimes != nil {
			stTime = time.Now()
		}
		if !do(i) {
			break
		}
		oneTimes(i, int(time.Since(stTime).Milliseconds()))
	}
}

func (m *Model) Trains(trainX, trainY []*mat.Dense) {
	m.TrainTimes(trainX, trainY, nil)
}

func (m *Model) TrainTimes(trainX, trainY []*mat.Dense, oneTimes func(int, int)) {
	WithTimes(len(trainX), func(i int) bool {
		x, y := trainX[i], trainY[i]
		m.Train(x, y)
		return !m.IsDone()
	}, oneTimes)
}
