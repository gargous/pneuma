package nn

import (
	"pneuma/common"

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

func (m *Model) Copy(src *Model) {
	m.layers = make([]*layer, len(src.layers))
	for k, l := range src.layers {
		m.layers[k] = &layer{}
		m.layers[k].copy(l)
	}
	m.loss = &loss{}
	m.loss.copy(src.loss)
}

func (m *Model) SetTarget(tar common.ITarget, param *LossParam) {
	m.loss = &loss{
		target: tar,
		param:  param,
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
	m.layers[idx] = &layer{
		optimizer: opt,
		hlayers:   layers,
	}
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

func (m *Model) Predict(x *mat.Dense) *mat.Dense {
	a := x
	for i := 0; i < len(m.layers); i++ {
		a = m.layers[i].predict(a)
	}
	return a
}

func (m *Model) Test(x, y *mat.Dense) (loss, acc float64) {
	pred := m.Predict(x)
	loss = m.Loss(pred, y)
	acc = m.Acc(pred, y)
	return
}

func (m *Model) Acc(pred, targ *mat.Dense) float64 {
	cnt := 0.0
	_, batch := targ.Dims()
	for j := 0; j < batch; j++ {
		pred := mat.Col(nil, j, pred)
		targ := mat.Col(nil, j, targ)
		maxPredIdx := floats.MaxIdx(pred)
		maxTargIdx := floats.MaxIdx(targ)
		if maxPredIdx == maxTargIdx {
			cnt++
		}
	}
	return cnt / float64(batch)
}

func (m *Model) Loss(pred, targ *mat.Dense) float64 {
	return m.loss.target.Loss(pred, targ)
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
		if oneTimes != nil {
			oneTimes(trainTimes)
			trainTimes++
		}
		if m.IsDone() {
			break
		}
	}
}
