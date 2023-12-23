package dnn

import (
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

type IniSAE struct {
	saeMods []*nn.Model
}

func NewIniSAE(m *nn.Model) *IniSAE {
	ini := &IniSAE{}
	ini.saeMods = make([]*nn.Model, m.LayerCnt()-1)
	for i := 0; i < m.LayerCnt()-1; i++ {
		opt, hlayers := m.Layer(i)
		tar := nn.NewTarMSE()
		param := nn.NewLossParam()
		saeM := nn.NewModel()
		saeM.AddLayer(opt, hlayers...)
		for j := 0; j < len(hlayers); j++ {
			if linear, ok := hlayers[j].(*nn.HLayerLinear); ok {
				r, c := linear.Dims()
				lay := nn.NewHLayerLinear()
				lay.InitSize([]int{c, r})
				saeM.AddLayer(nn.NewOptNormal(0.001), lay)
				break
			}
		}

		saeM.SetTarget(tar, param)
		ini.saeMods[i] = saeM
	}
	return ini
}

func (ini *IniSAE) Init(trainX []*mat.Dense) {
	for _, m := range ini.saeMods {
		var xs []*mat.Dense
		for i := 0; i < len(trainX); i++ {
			x := trainX[i]
			m.Train(x, x)
			if m.IsDone() {
				break
			}
			_, l := m.Layer(0)
			xs = append(xs, l[0].(*nn.HLayerLinear).Y)
		}
		trainX = xs
	}
}
