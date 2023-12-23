package cu

import (
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

func DevideOptData(cal *MatCaltor, datas, deltas []mat.Matrix) (datasInHost, deltasInHost, datasInDevice, deltasInDevice []mat.Matrix) {
	for i := 0; i < len(datas); i++ {
		data := datas[i]
		delta := deltas[i]
		if cal.Idx(data) >= 0 {
			datasInDevice = append(datasInDevice, data)
			deltasInDevice = append(deltasInDevice, delta)
		} else {
			datasInHost = append(datasInHost, data)
			deltasInHost = append(deltasInHost, delta)
		}
	}
	return
}

type OptNormal struct {
	*nn.OptNormal
	cal *MatCaltor
}

func NewOptNormal(cal *MatCaltor, lr float64) *OptNormal {
	return &OptNormal{
		OptNormal: nn.NewOptNormal(lr),
		cal:       cal,
	}
}

func (opt *OptNormal) Update(datas, deltas []mat.Matrix) {
	lr := opt.Param()
	daH, deH, daD, deD := DevideOptData(opt.cal, datas, deltas)
	for i := 0; i < len(daD); i++ {
		opt.cal.AddScaled(daD[i], -lr, deD[i])
	}
	opt.OptNormal.Update(daH, deH)
}
