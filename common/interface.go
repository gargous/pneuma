package common

import "gonum.org/v1/gonum/mat"

type IHLayer interface {
	Forward(x *mat.Dense) (y *mat.Dense)
	Backward(dy *mat.Dense) (dx *mat.Dense)
}

type IHLayerOptimizer interface {
	IHLayer
	Optimize() (datas, deltas []mat.Matrix)
}

type IHLayerPredictor interface {
	IHLayer
	Predict(x *mat.Dense) (y *mat.Dense)
}

type IHLayerSizeIniter interface {
	IHLayer
	InitSize([]int) []int
}

type IOptimizer interface {
	Update(datas, deltas []mat.Matrix)
}

type IOptimizerCoLayer interface {
	IOptimizer
	SetIHLayers(ls ...IHLayerOptimizer)
}

type ITarget interface {
	Acc(pred, targ *mat.Dense) (acc float64)
	Loss(pred, targ *mat.Dense) (y float64)
	LossEach(pred, targ *mat.Dense) (loss *mat.Dense)
	Backward() (dy *mat.Dense)
}
