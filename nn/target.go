package nn

import "gonum.org/v1/gonum/mat"

type ITarget interface {
	Loss(ret, tar *mat.Dense) float64
	Backward(ret, tar *mat.Dense) (dy *mat.Dense)
	Param() *TargetParam
}

type TargetParam struct {
	Threshold float64
	MinLoss   float64
	MinTimes  int
}

type TargetMSE struct {
	param *TargetParam
}
