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

func (m *Model) Predicts(predX []*mat.Dense) []*mat.Dense {
	predY := make([]*mat.Dense, len(predX))
	for i := 0; i < len(predX); i++ {
		predY[i] = m.Predict(predX[i])
	}
	return predY
}

func (m *Model) Accs(preds, targs []*mat.Dense) float64 {
	cnt := 0.0
	acnt := 0.0
	for i := 0; i < len(preds); i++ {
		_, batch := targs[i].Dims()
		acc := m.Acc(preds[i], targs[i])
		cnt += acc * float64(batch)
		acnt += float64(batch)
	}
	return cnt / acnt
}

func (m *Model) MeanLosses(preds, targs []*mat.Dense) float64 {
	loss := 0.0
	for i := 0; i < len(preds); i++ {
		loss += m.Loss(preds[i], targs[i])
	}
	return loss / float64(len(preds))
}

func (m *Model) Trains(trainX, trainY []*mat.Dense) {
	m.TrainTimes(trainX, trainY, nil)
}

func (m *Model) TrainTimes(trainX, trainY []*mat.Dense, oneTimes func(int)) {
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
