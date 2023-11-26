package nn

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type ModelBuilder struct {
	size      []int
	layers    []func(r, c int) IHLayer
	optimizer func(r, c int) IOptimizer
	tar       func() ITarget
}

func NewModelBuilder() *ModelBuilder {
	return &ModelBuilder{}
}

func (m *ModelBuilder) Size(size ...int) {
	m.size = size
}

func (m *ModelBuilder) Layer(layer func() IHLayer) {
	m.layers = append(m.layers, func(r, c int) IHLayer { return layer() })
}

func (m *ModelBuilder) LayerAt(layer func(r, c int) IHLayer) {
	m.layers = append(m.layers, layer)
}

func (m *ModelBuilder) Optimizer(opt func() IOptimizer) {
	m.optimizer = func(r, c int) IOptimizer { return opt() }
}

func (m *ModelBuilder) OptimizerAt(opt func(r, c int) IOptimizer) {
	m.optimizer = opt
}

func (m *ModelBuilder) Target(tar func() ITarget) {
	m.tar = tar
}

func (m *ModelBuilder) Build() *Model {
	model := NewModel()
	for i := 0; i < len(m.size)-1; i++ {
		c := m.size[i]
		r := m.size[i+1]

		opt := m.optimizer(r, c)
		hlayers := []IHLayer{NewHLayerLinear(r, c)}
		for _, hlayer := range m.layers {
			hlayers = append(hlayers, hlayer(r, c))
		}
		model.AddLayer(
			opt,
			hlayers...,
		)
	}
	model.SetTarget(m.tar(), NewLossParam())
	return model
}

type Model struct {
	layers []*layer
	loss   *loss
}

func NewModel() *Model {
	return &Model{}
}

func (m *Model) Copy(src *Model) {
	m.layers = make([]*layer, len(src.layers))
	for k, l := range src.layers {
		m.layers[k] = &layer{}
		m.layers[k].copy(l)
	}
	m.loss = &loss{}
	m.loss.copy(src.loss)
}

func (m *Model) SetTarget(tar ITarget, param *LossParam) {
	m.loss = &loss{
		target: tar,
		param:  param,
	}
}

func (m *Model) AddLayer(opt IOptimizer, layers ...IHLayer) {
	m.layers = append(m.layers, &layer{
		optimizer: opt,
		hlayers:   layers,
	})
}

func (m *Model) Train(x, y *mat.Dense) {
	a := x
	for i := 0; i < len(m.layers); i++ {
		a = m.layers[i].forward(a)
	}
	if m.loss.check(a, y) {
		return
	}
	da := m.loss.backward()
	for i := len(m.layers) - 1; i >= 0; i-- {
		da = m.layers[i].backward(da)
	}
	for i := 0; i < len(m.layers); i++ {
		m.layers[i].update()
	}
}

func (m *Model) Test(x, y *mat.Dense) float64 {
	cnt := 0.0
	a := x
	for i := 0; i < len(m.layers); i++ {
		a = m.layers[i].predict(a)
	}
	_, batch := y.Dims()
	preds := a
	for j := 0; j < batch; j++ {
		pred := mat.Col(nil, j, preds)
		targ := mat.Col(nil, j, y)
		maxPredIdx := floats.MaxIdx(pred)
		maxTargIdx := floats.MaxIdx(targ)
		if maxPredIdx == maxTargIdx {
			cnt++
		}
	}
	return cnt / float64(batch)
}

func (m *Model) LossPopMean() float64 {
	mean := stat.Mean(m.loss.losses, nil)
	m.loss.losses = nil
	return mean
}

func (m *Model) LossMean() float64 {
	return stat.Mean(m.loss.losses, nil)
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

func (m *Model) TrainEpoch(trainX, trainY []*mat.Dense) {
	m.TrainEpochTimes(trainX, trainY, nil)
}

func (m *Model) TrainEpochTimes(trainX, trainY []*mat.Dense, oneTimes func(int)) {
	var trainTimes int
	for i := 0; i < len(trainX); i++ {
		x, y := trainX[i], trainY[i]
		m.Train(x, y)
		if m.IsDone() {
			break
		}
		if oneTimes != nil {
			oneTimes(trainTimes)
			trainTimes++
		}
	}
}
