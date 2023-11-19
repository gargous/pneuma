package nn

import (
	"gonum.org/v1/gonum/mat"
)

type layer struct {
	hlayers   []IHLayer
	optimizer IOptimizer
}

func (l *layer) forward(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(l.hlayers); i++ {
		a = l.hlayers[i].Forward(a)
	}
	return a
}

func (l *layer) predict(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(l.hlayers); i++ {
		predictor, isPredictor := l.hlayers[i].(IHLayerPredictor)
		if isPredictor {
			a = predictor.Predict(a)
		} else {
			a = l.hlayers[i].Forward(a)
		}
	}
	return a
}

func (l *layer) backward(da *mat.Dense) *mat.Dense {
	for i := len(l.hlayers) - 1; i >= 0; i-- {
		da = l.hlayers[i].Backward(da)
	}
	return da
}

func (l *layer) update() {
	var optDatas []mat.Matrix
	var optDeltas []mat.Matrix
	for i := 0; i < len(l.hlayers); i++ {
		opt, isOpt := l.hlayers[i].(IHLayerOptimizer)
		if !isOpt {
			continue
		}
		optData, optDelta := opt.Optimize()
		optDatas = append(optDatas, optData...)
		optDeltas = append(optDeltas, optDelta...)
	}
	l.optimizer.Update(optDatas, optDeltas)
}
