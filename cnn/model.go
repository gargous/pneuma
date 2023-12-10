package cnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelBuilder struct {
	inpSize      []int
	convLayers   []*HLayerConv
	fullConnSize []int
	clayers      []func() common.IHLayer
	flayers      []func() common.IHLayer
	optimizer    func() common.IOptimizer
	tar          func() common.ITarget
}

func NewModelBuilder(inpSize []int) *ModelBuilder {
	return &ModelBuilder{
		inpSize: inpSize,
	}
}

func (m *ModelBuilder) Conv(core, stride []int, padding bool) {
	var newLayer *HLayerConv
	if len(m.convLayers) == 0 {
		newLayer = NewHLayerConv(m.inpSize, core, stride, padding)
	} else {
		inpSize := m.convLayers[len(m.convLayers)-1].ouptSize
		newLayer = NewHLayerConv(inpSize, core, stride, padding)
	}
	m.convLayers = append(m.convLayers, newLayer)
}

func (m *ModelBuilder) FSize(size ...int) {
	m.fullConnSize = size
}

func (m *ModelBuilder) CLayer(layer func() common.IHLayer) {
	m.clayers = append(m.clayers, layer)
}

func (m *ModelBuilder) FLayer(layer func() common.IHLayer) {
	m.flayers = append(m.flayers, layer)
}

func (m *ModelBuilder) Optimizer(opt func() common.IOptimizer) {
	m.optimizer = func() common.IOptimizer { return opt() }
}

func (m *ModelBuilder) Target(tar func() common.ITarget) {
	m.tar = tar
}

func (m *ModelBuilder) Build() *nn.Model {
	model := nn.NewModel()
	for i := 0; i < len(m.convLayers); i++ {
		opt := m.optimizer()
		hlayers := []common.IHLayer{m.convLayers[i]}
		for _, hlayer := range m.clayers {
			hlayers = append(hlayers, hlayer())
		}
		model.AddLayer(
			opt,
			hlayers...,
		)
	}
	fSize := intsProd(m.convLayers[len(m.convLayers)-1].ouptSize)
	m.fullConnSize = append([]int{fSize}, m.fullConnSize...)
	for i := 0; i < len(m.fullConnSize)-1; i++ {
		c := m.fullConnSize[i]
		r := m.fullConnSize[i+1]

		opt := m.optimizer()
		hlayers := []common.IHLayer{nn.NewHLayerLinear(r, c)}
		for _, hlayer := range m.flayers {
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
