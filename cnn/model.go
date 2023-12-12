package cnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelConvBuilder struct {
	conv   func(inpSize []int) (l common.IHLayer, oupSize []int)
	layers []func(inpSize []int) (l common.IHLayer, oupSize []int)
}

func (m *ModelConvBuilder) CLayer(layer func(inpSize []int) (l common.IHLayer, oupSize []int)) {
	m.layers = append(m.layers, func(inpSize []int) (l common.IHLayer, oupSize []int) {
		l, oupSize = layer(inpSize)
		if oupSize == nil {
			oupSize = inpSize
		}
		return
	})
}

func (m *ModelConvBuilder) Build(inpSize []int) (ls []common.IHLayer, oupSize []int) {
	ls = make([]common.IHLayer, len(m.layers)+1)
	ls[0], oupSize = m.conv(inpSize)
	for i := 0; i < len(m.layers); i++ {
		ls[i+1], oupSize = m.layers[i](oupSize)
	}
	return
}

type ModelBuilder struct {
	inpSize      []int
	fullConnSize []int
	clayers      []func(inpSize []int) (l common.IHLayer, oupSize []int)
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

func (m *ModelBuilder) Conv(core, stride []int, padding bool) *ModelConvBuilder {
	cb := &ModelConvBuilder{}
	cb.conv = func(inpSize []int) (l common.IHLayer, oupSize []int) {
		conv := NewHLayerConv(inpSize, core, stride, padding)
		return conv, conv.ouptSize
	}
	return cb
}

func (m *ModelBuilder) FSize(size ...int) {
	m.fullConnSize = size
}

func (m *ModelBuilder) CLayer(layer func(inpSize []int) (l common.IHLayer, oupSize []int)) {
	m.clayers = append(m.clayers, func(inpSize []int) (l common.IHLayer, oupSize []int) {
		l, oupSize = layer(inpSize)
		if oupSize == nil {
			oupSize = inpSize
		}
		return
	})
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
		cls, oupSize := cb.Build(inpSize)
		for _, clayer := range m.clayers {
			cl, size := clayer(oupSize)
			cls = append(cls, cl)
			oupSize = size
		}
		model.AddLayer(
			opt,
			cls...,
		)
		inpSize = oupSize
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
