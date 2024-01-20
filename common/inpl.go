package common

import "gonum.org/v1/gonum/mat"

func OptimizeData(layers ...IHLayer) (optDatas []mat.Matrix, optDeltas []mat.Matrix) {
	for i := 0; i < len(layers); i++ {
		opt, isOpt := layers[i].(IHLayerOptimizer)
		if !isOpt {
			continue
		}
		optData, optDelta := opt.Optimize()
		optDatas = append(optDatas, optData...)
		optDeltas = append(optDeltas, optDelta...)
	}
	return
}

func Predic(layer IHLayer, x *mat.Dense) *mat.Dense {
	predictor, isPredictor := layer.(IHLayerPredictor)
	if isPredictor {
		return predictor.Predict(x)
	} else {
		return layer.Forward(x)
	}
}
