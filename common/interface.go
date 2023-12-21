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

type IOptimizer interface {
	Update(datas, deltas []mat.Matrix)
}

type ITarget interface {
	Loss(pred, targ *mat.Dense) (y float64)
	Backward() (dy *mat.Dense)
}

type IConvCaltor interface {
	Forward(packX, packY, w, b *mat.Dense)
	Backward(packDx, packDy, dw, db *mat.Dense)
}
