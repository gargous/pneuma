package cnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelMiniBuilder struct {
	lays []common.IHLayer
	opt  common.IOptimizer
}

func (m *ModelMiniBuilder) Lay(l common.IHLayer) {
	m.lays = append(m.lays, l)
}

func (m *ModelMiniBuilder) Opt(l common.IOptimizer) {
	m.opt = l
}

func (m *ModelMiniBuilder) Build(cb func(*ModelMiniBuilder)) {
	cb(m)
}

type ModelBuilder struct {
	csize []int
	convs []*ModelMiniBuilder
	clays []func() common.IHLayer
	copt  func() common.IOptimizer
	fsize []int
	fulls []*ModelMiniBuilder
	flays []func() common.IHLayer
	fopt  func() common.IOptimizer
	tar   common.ITarget
}

func NewModelBuilder(csize, fsize []int) *ModelBuilder {
	return &ModelBuilder{
		csize: csize,
		fsize: fsize,
	}
}
func (m *ModelBuilder) F() *ModelMiniBuilder {
	b := &ModelMiniBuilder{}
	m.fulls = append(m.fulls, b)
	return b
}

func (m *ModelBuilder) C() *ModelMiniBuilder {
	b := &ModelMiniBuilder{}
	m.convs = append(m.convs, b)
	return b
}

func (m *ModelBuilder) CLay(l func() common.IHLayer) {
	m.clays = append(m.clays, l)
}

func (m *ModelBuilder) FLay(l func() common.IHLayer) {
	m.flays = append(m.flays, l)
}

func (m *ModelBuilder) COpt(l func() common.IOptimizer) {
	m.copt = l
}

func (m *ModelBuilder) FOpt(l func() common.IOptimizer) {
	m.fopt = l
}

func (m *ModelBuilder) Tar(l common.ITarget) {
	m.tar = l
}

func (m *ModelBuilder) Build() *nn.Model {
	model := nn.NewModel()
	csize := m.csize
	for _, conv := range m.convs {
		for _, layFun := range m.clays {
			conv.lays = append(conv.lays, layFun())
		}
		for _, lay := range conv.lays {
			initer, ok := lay.(common.IHLayerSizeIniter)
			if ok {
				csize = initer.InitSize(csize)
			}
		}
		if conv.opt == nil {
			model.AddLayer(m.copt(), conv.lays...)
		} else {
			model.AddLayer(conv.opt, conv.lays...)
		}
	}
	fsize := append([]int{common.IntsProd(csize)}, m.fsize...)
	rest := len(m.fsize) - len(m.fulls)
	for i := 0; i < rest; i++ {
		m.fulls = append(m.fulls, &ModelMiniBuilder{})
	}
	for i, full := range m.fulls {
		size := []int{fsize[i+1], fsize[i]}
		for _, layFun := range m.flays {
			full.lays = append(full.lays, layFun())
		}
		for _, lay := range full.lays {
			initer, ok := lay.(common.IHLayerSizeIniter)
			if ok {
				initer.InitSize(size)
			}
		}
		if full.opt == nil {
			model.AddLayer(m.fopt(), full.lays...)
		} else {
			model.AddLayer(full.opt, full.lays...)
		}
	}
	model.SetTarget(m.tar, nn.NewLossParam())
	return model
}
