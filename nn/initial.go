package nn

import (
	"gonum.org/v1/gonum/mat"
)

type IniStrategy interface {
	Init(trainX []*mat.Dense)
}

type iniSAE struct {
	saeMods []*Model
}

func NewIniSAE(m *Model) IniStrategy {
	ini := &iniSAE{}
	ini.saeMods = make([]*Model, len(m.layers)-1)
	for i := 0; i < len(m.layers)-1; i++ {
		saeM := &Model{}
		saeM.Copy(m)
		oneLayer := saeM.layers[i]
		saeM.layers = make([]*layer, 2)
		saeM.layers[0] = oneLayer
		saeM.layers[1] = &layer{}
		saeM.layers[1].copy(oneLayer)
		r, c := saeM.layers[1].linearHLayer().w.Dims()
		saeM.layers[1].reshapeAsNew(c, r)
		ini.saeMods[i] = saeM
	}
	return ini
}

func (ini *iniSAE) Init(trainX []*mat.Dense) {
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
