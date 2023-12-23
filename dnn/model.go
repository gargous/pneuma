package dnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelBuilder struct {
	size      []int
	layers    []func() common.IHLayer
	optimizer func() common.IOptimizer
	tar       func() common.ITarget
}

func NewModelBuilder() *ModelBuilder {
	return &ModelBuilder{}
}

func (m *ModelBuilder) Size(size ...int) {
	m.size = size
}

func (m *ModelBuilder) Layer(layer func() common.IHLayer) {
	m.layers = append(m.layers, func() common.IHLayer { return layer() })
}

func (m *ModelBuilder) Optimizer(opt func() common.IOptimizer) {
	m.optimizer = opt
}

func (m *ModelBuilder) Target(tar func() common.ITarget) {
	m.tar = tar
}

func (m *ModelBuilder) Build() *nn.Model {
	model := nn.NewModel()
	for i := 0; i < len(m.size)-1; i++ {
		c := m.size[i]
		r := m.size[i+1]

		opt := m.optimizer()
		lay := nn.NewHLayerLinear()
		lay.InitSize([]int{r, c})
		hlayers := []common.IHLayer{lay}
		for _, hlayer := range m.layers {
			hlayers = append(hlayers, hlayer())
		}
		model.AddLayer(
			opt,
			hlayers...,
		)
	}
	model.SetTarget(m.tar(), nn.NewLossParam())
	return model
}
