package nn

import (
	"pneuma/common"

	"gonum.org/v1/gonum/mat"
)

type IniSAE struct {
	saeMods []*Model
}

func NewIniSAE(m *Model) *IniSAE {
	ini := &IniSAE{}
	ini.saeMods = make([]*Model, m.LayerCnt()-1)
	for i := 0; i < m.LayerCnt()-1; i++ {
		opt, hlayers := m.Layer(i)
		tar, param := m.Target()

		saeM := NewModel()

		newParam := &LossParam{}
		newParam.Copy(param)
		saeM.SetTarget(common.CopyITarget(tar), newParam)

		hls0 := make([]common.IHLayer, len(hlayers))
		for j := 0; j < len(hlayers); j++ {
			hls0[j] = common.CopyIHLayer(hlayers[j])
		}
		saeM.AddLayer(common.CopyIOptimizer(opt), hls0...)
		hls1 := make([]common.IHLayer, len(hlayers))
		for j := 0; j < len(hlayers); j++ {
			hls1[j] = common.CopyIHLayer(hlayers[j])
			if linear, isLinear := hls1[j].(*HLayerLinear); isLinear {
				r, c := linear.Dims()
				hls1[j] = NewHLayerLinear(c, r)
			}
		}
		saeM.AddLayer(common.CopyIOptimizer(opt), hls1...)
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
			xs = append(xs, m.layers[0].ret)
		}
		trainX = xs
	}
}
