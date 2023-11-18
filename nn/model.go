package nn

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	layers []*layer
	target ITarget
	losses []float64
}

func (m *Model) SetTarget(tar ITarget) {
	m.target = tar
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
	loss := m.target.Loss(a, y)
	m.losses = append(m.losses, loss)
	if m.isLossDone() {
		return
	}
	da := m.target.Backward(a, y)
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

func (m *Model) isLossDone() bool {
	param := m.target.Param()
	if param == nil {
		return false
	}
	if len(m.losses) <= param.MinTimes {
		return false
	}
	lossCur := m.losses[len(m.losses)-1]
	lossOld := m.losses[len(m.losses)-2]
	if math.Abs(lossCur-lossOld) > param.Threshold {
		return false
	}
	return lossCur <= param.MinLoss
}
