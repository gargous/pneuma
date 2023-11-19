package nn

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type Model struct {
	layers []*layer
	loss   *loss
}

func NewModel() *Model {
	return &Model{}
}

func NewStdModel(size []int, layers func(r, c int) (IOptimizer, []IHLayer)) *Model {
	m := &Model{}
	for i := 0; i < len(size)-1; i++ {
		c := size[i]
		r := size[i+1]
		opt, hlayers := layers(r, c)
		linear := NewHLayerLinear(r, c)
		hlayers = append([]IHLayer{linear}, hlayers...)
		m.AddLayer(
			opt,
			hlayers...,
		)
	}
	return m
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

func (m *Model) LossDrop() {
	m.loss.losses = nil
}

func (m *Model) LossMean() float64 {
	return stat.Mean(m.loss.losses, nil)
}

func (m *Model) IsDone() bool {
	return m.loss.isDone()
}
