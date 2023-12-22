package cu

import (
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

type OptNormal struct {
	*nn.OptNormal
	calters []IMatCaltor
}

func NewOptNormal(lr float64) *OptNormal {
	return &OptNormal{
		OptNormal: nn.NewOptNormal(lr),
	}
}

func (opt *OptNormal) SetIHLayers(ls ...common.IHLayerOptimizer) {
	opt.calters = make([]IMatCaltor, 0)
	for i := 0; i < len(ls); i++ {
		caltor, ok := ls[i].(IMatCaltor)
		if ok {
			opt.calters = append(opt.calters, caltor)
		}
	}
}

func (opt *OptNormal) FindCalterIdx(data mat.Matrix) int {
	for i := 0; i < len(opt.calters); i++ {
		if opt.calters[i].Idx(data) >= 0 {
			return i
		}
	}
	return -1
}

func (opt *OptNormal) Update(datas, deltas []mat.Matrix) {
	datasIndevice := make([][]mat.Matrix, len(opt.calters))
	deltasIndevice := make([][]mat.Matrix, len(opt.calters))
	datasInHost := make([]mat.Matrix, 0)
	deltasInHost := make([]mat.Matrix, 0)
	for i := 0; i < len(datas); i++ {
		caltorIdx := opt.FindCalterIdx(datas[i])
		if caltorIdx >= 0 {
			datasIndevice[caltorIdx] = append(datasIndevice[caltorIdx], datas[i])
			deltasIndevice[caltorIdx] = append(deltasIndevice[caltorIdx], deltas[i])
		} else {
			datasInHost = append(datasInHost, datas[i])
			deltasInHost = append(deltasInHost, deltas[i])
		}
	}
	lr := opt.Param()
	for i := 0; i < len(datasIndevice); i++ {
		caltor := opt.calters[i]
		datas := datasIndevice[i]
		deltas := deltasIndevice[i]
		for j := 0; j < len(datas); j++ {
			caltor.AddScaled(datas[j], -lr, deltas[j])
		}
	}
	opt.OptNormal.Update(datasInHost, deltasInHost)
}
