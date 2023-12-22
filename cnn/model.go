package cnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelConvBuilder struct {
	layers []func(inpSize []int) common.IHLayer
}

func (m *ModelConvBuilder) CLayer(layer func(inpSize []int) common.IHLayer) *ModelConvBuilder {
	m.layers = append(m.layers, layer)
	return m
}

type ModelBuilder struct {
	inpSize      []int
	fullConnSize []int
	clayers      []func(inpSize []int) common.IHLayer
	flayers      []func() common.IHLayer
	optimizer    func() common.IOptimizer
	tar          func() common.ITarget
	cbuilders    []*ModelConvBuilder
}

func NewModelBuilder(inpSize []int) *ModelBuilder {
	return &ModelBuilder{
		inpSize: inpSize,
	}
}

func (m *ModelBuilder) ConvStd(core, stride []int, padding bool) *ModelConvBuilder {
	cb := &ModelConvBuilder{}
	cb.layers = []func(inpSize []int) common.IHLayer{func(inpSize []int) common.IHLayer {
		return NewHLayerConv(inpSize, core, stride, padding)
	}}
	m.cbuilders = append(m.cbuilders, cb)
	return cb
}

func (m *ModelBuilder) Conv(layer func(inpSize []int) common.IHLayer) *ModelConvBuilder {
	cb := &ModelConvBuilder{}
	cb.layers = []func(inpSize []int) common.IHLayer{layer}
	m.cbuilders = append(m.cbuilders, cb)
	return cb
}

func (m *ModelBuilder) FSize(size ...int) {
	m.fullConnSize = size
}

func (m *ModelBuilder) CLayer(layer func(inpSize []int) common.IHLayer) {
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
	inpSize := m.inpSize
	for i := 0; i < len(m.cbuilders); i++ {
		opt := m.optimizer()
		cb := m.cbuilders[i]
		var cls []common.IHLayer
		for _, clayer := range append(cb.layers, m.clayers...) {
			cl := clayer(inpSize)
			if oupt, isOupt := cl.(iHLayerOupt); isOupt {
				inpSize = oupt.OuptSize()
			}
			cls = append(cls, cl)
		}
		model.AddLayer(
			opt,
			cls...,
		)
	}
	m.fullConnSize = append([]int{intsProd(inpSize)}, m.fullConnSize...)
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
