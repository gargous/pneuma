package nn

import (
	"gonum.org/v1/gonum/mat"
)

type layer struct {
	hlayers   []IHLayer
	ret       *mat.Dense
	optimizer IOptimizer
}

func (l *layer) copy(src *layer) {
	l.optimizer = CopyIOptimizer(src.optimizer)
	l.hlayers = make([]IHLayer, len(src.hlayers))
	if src.ret != nil {
		l.ret = mat.DenseCopyOf(src.ret)
	}
	for k, srcHL := range src.hlayers {
		l.hlayers[k] = CopyIHLayer(srcHL)
	}
}

func (l *layer) linearHLayer() *HLayerLinear {
	for i := 0; i < len(l.hlayers); i++ {
		if hl, ok := l.hlayers[i].(*HLayerLinear); ok {
			return hl
		}
	}
	return nil
}

func (l *layer) reshapeAsNew(r, c int) {
	for i := 0; i < len(l.hlayers); i++ {
		if resh, ok := l.hlayers[i].(IHLayerReshape); ok {
			resh.ReshapeAsNew(r, c)
		}
	}
}

func (l *layer) forward(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(l.hlayers); i++ {
		a = l.hlayers[i].Forward(a)
	}
	l.ret = a
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
