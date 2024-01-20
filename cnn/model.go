package cnn

import (
	"pneuma/common"
	"pneuma/nn"
)

type ModelSizeBuilder struct {
	Size    []int
	Uniques []*nn.ModelSample
	*nn.ModelBuilder
}

func NewModelSizeBuilder(size []int) *ModelSizeBuilder {
	return &ModelSizeBuilder{
		Size:         size,
		ModelBuilder: &nn.ModelBuilder{},
	}
}

func (b *ModelSizeBuilder) One() *nn.ModelSample {
	s := &nn.ModelSample{}
	b.Uniques = append(b.Uniques, s)
	return s
}

func (b *ModelSizeBuilder) BulldWithSize(model *nn.Model, cb func(initer common.IHLayerSizeIniter, i int, size []int) []int) []int {
	size := b.Size
	for i, conv := range b.Uniques {
		for _, layFun := range b.Lays {
			conv.Lays = append(conv.Lays, layFun())
		}
		for _, lay := range conv.Lays {
			initer, ok := lay.(common.IHLayerSizeIniter)
			if ok {
				size = cb(initer, i, size)
			}
		}
		if conv.Optimizer == nil {
			model.AddLayer(b.Optimizer(), conv.Lays...)
		} else {
			model.AddLayer(conv.Optimizer, conv.Lays...)
		}
	}
	return size
}

func (b *ModelSizeBuilder) Bulld(model *nn.Model) []int {
	return b.BulldWithSize(model, func(initer common.IHLayerSizeIniter, i int, size []int) []int {
		return initer.InitSize(size)
	})
}

type ModelBuilder struct {
	c   *ModelSizeBuilder
	f   *ModelSizeBuilder
	tar common.ITarget
}

func NewModelBuilder(csize, fsize []int) *ModelBuilder {
	return &ModelBuilder{
		c: NewModelSizeBuilder(csize),
		f: NewModelSizeBuilder(fsize),
	}
}
func (m *ModelBuilder) F(cb func(*nn.ModelSample)) {
	m.f.One().Use(cb)
}

func (m *ModelBuilder) C(cb func(*nn.ModelSample)) {
	m.c.One().Use(cb)
}

func (m *ModelBuilder) CLay(l func() common.IHLayer) {
	m.c.Lay(l)
}

func (m *ModelBuilder) FLay(l func() common.IHLayer) {
	m.f.Lay(l)
}

func (m *ModelBuilder) COpt(l func() common.IOptimizer) {
	m.c.Opt(l)
}

func (m *ModelBuilder) FOpt(l func() common.IOptimizer) {
	m.f.Opt(l)
}

func (m *ModelBuilder) Tar(l common.ITarget) {
	m.tar = l
}

func (m *ModelBuilder) Build() *nn.Model {
	model := nn.NewModel()
	csize := m.c.Bulld(model)
	fsize := append([]int{common.IntsProd(csize)}, m.f.Size...)
	rest := len(m.f.Size) - len(m.f.Uniques)
	for i := 0; i < rest; i++ {
		m.f.Uniques = append(m.f.Uniques, &nn.ModelSample{})
	}
	m.f.BulldWithSize(model, func(initer common.IHLayerSizeIniter, i int, size []int) []int {
		size = []int{fsize[i+1], fsize[i]}
		initer.InitSize(size)
		return size
	})
	model.SetTarget(m.tar, nn.NewLossParam())
	return model
}
