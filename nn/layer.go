package nn

import (
	"pneuma/common"

	"gonum.org/v1/gonum/mat"
)

type layer struct {
	hlayers   []common.IHLayer
	optimizer common.IOptimizer
}

func (l *layer) forward(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(l.hlayers); i++ {
		a = l.hlayers[i].Forward(a)
	}
	return a
}

func (l *layer) predict(a *mat.Dense) *mat.Dense {
	for i := 0; i < len(l.hlayers); i++ {
		a = common.Predic(l.hlayers[i], a)
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
	l.optimizer.Update(common.OptimizeData(l.hlayers...))
}
