package frcnn

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Bounds struct {
	areas *mat.VecDense
	maxs  *mat.Dense
	mins  *mat.Dense
	subs  *mat.Dense
	ctrs  *mat.Dense
}

func NewBounds(r, c int) *Bounds {
	ret := &Bounds{}
	ret.maxs = mat.NewDense(r, c, nil)
	ret.mins = mat.NewDense(r, c, nil)
	return ret
}

func NewBoxs(r, c int) *Bounds {
	ret := &Bounds{}
	ret.ctrs = mat.NewDense(r, c, nil)
	ret.subs = mat.NewDense(r, c, nil)
	return ret
}

func NewEyeBounds(r, c int) *Bounds {
	ret := NewBounds(r, c)
	ret.maxs.Apply(func(i, j int, v float64) float64 {
		return 0.5
	}, ret.maxs)
	ret.mins.Apply(func(i, j int, v float64) float64 {
		return -0.5
	}, ret.mins)
	return ret
}

func (b *Bounds) SetAll(bnd *mat.Dense) {
	cnt, size := b.Dims()
	b.mins.Copy(bnd.Slice(0, cnt, 0, size))
	b.maxs.Copy(bnd.Slice(0, cnt, size, size*2))
}

func (b *Bounds) SetRow(i int, row *mat.VecDense) {
	size := b.Size()
	minVec := row.SliceVec(0, size).(*mat.VecDense)
	maxVec := row.SliceVec(size, size*2).(*mat.VecDense)
	b.maxs.RowView(i).(*mat.VecDense).CopyVec(maxVec)
	b.mins.RowView(i).(*mat.VecDense).CopyVec(minVec)
}

func (b *Bounds) Centers() *mat.Dense {
	if b.ctrs == nil {
		r, c := b.mins.Dims()
		b.ctrs = mat.NewDense(r, c, nil)
		b.ctrs.Add(b.mins, b.maxs)
		b.ctrs.Scale(0.5, b.ctrs)
	}
	return b.ctrs
}

func (b *Bounds) Subs() *mat.Dense {
	if b.subs == nil {
		r, c := b.mins.Dims()
		b.subs = mat.NewDense(r, c, nil)
		b.subs.Sub(b.maxs, b.mins)
	}
	return b.subs
}

func (b *Bounds) Areas() *mat.VecDense {
	if b.areas == nil {
		r, _ := b.mins.Dims()
		b.areas = mat.NewVecDense(r, nil)
		subs := b.Subs()
		for i := 0; i < r; i++ {
			subVec := subs.RowView(i)
			b.areas.SetVec(i, mat.Dot(subVec, subVec))
		}
	}
	return b.areas
}

func (b *Bounds) Len() int {
	r, _ := b.mins.Dims()
	return r
}

func (b *Bounds) Size() int {
	_, c := b.mins.Dims()
	return c
}

func (b *Bounds) iouAt(o *Bounds, i, j int, maxMinVec, minMaxVec, selects *mat.VecDense, selectPair []float64) float64 {
	aSize := maxMinVec.Len()
	aMin, aMax, aArea := b.mins.RowView(i), b.maxs.RowView(i), b.Areas().AtVec(i)
	bMin, bMax, bArea := o.mins.RowView(j), o.maxs.RowView(j), o.Areas().AtVec(j)
	for k := 0; k < aSize; k++ {
		selectPair[0] = aMin.AtVec(k)
		selectPair[1] = bMin.AtVec(k)
		maxIdx := floats.MaxIdx(selectPair)
		maxMinVec.SetVec(k, selectPair[maxIdx])
		selectPair[0] = aMax.AtVec(k)
		selectPair[1] = bMax.AtVec(k)
		minIdx := floats.MinIdx(selectPair)
		minMaxVec.SetVec(k, selectPair[minIdx])
		if selects != nil {
			selects.SetVec(k*2, float64(maxIdx))
			selects.SetVec(k*2+1, float64(minIdx))
		}
	}
	cArea := mat.Dot(maxMinVec, minMaxVec)
	uArea := aArea + bArea - cArea
	if uArea == 0 {
		return 0
	}
	return cArea / uArea
}

func (b *Bounds) IOUElem(o *Bounds) (ious *mat.VecDense, selects *mat.Dense) {
	aSize := b.Size()
	ious = mat.NewVecDense(b.Len(), nil)
	maxMinVec := mat.NewVecDense(aSize, nil)
	minMaxVec := mat.NewVecDense(aSize, nil)
	selects = mat.NewDense(b.Len(), aSize*2, nil)
	selectPair := make([]float64, aSize)
	for i := 0; i < b.Len(); i++ {
		ious.SetVec(i, b.iouAt(o, i, i, maxMinVec, minMaxVec, selects.RowView(i).(*mat.VecDense), selectPair))
	}
	return
}

func (b *Bounds) IOUCross(o *Bounds) (ious *mat.Dense) {
	aSize := b.Size()
	ious = mat.NewDense(b.Len(), o.Len(), nil)
	maxMinVec := mat.NewVecDense(aSize, nil)
	minMaxVec := mat.NewVecDense(aSize, nil)
	selectPair := make([]float64, aSize)
	ious.Apply(func(i, j int, v float64) float64 {
		return b.iouAt(o, i, j, maxMinVec, minMaxVec, nil, selectPair)
	}, ious)
	return
}

func (b *Bounds) SetByBox() {
	r, c := b.subs.Dims()
	halfSub := mat.NewDense(r, c, nil)
	halfSub.Scale(0.5, b.subs)
	b.maxs = mat.NewDense(r, c, nil)
	b.mins = mat.NewDense(r, c, nil)
	b.maxs.Add(b.ctrs, halfSub)
	b.mins.Sub(b.ctrs, halfSub)
}

func (b *Bounds) Dims() (r, c int) {
	return b.mins.Dims()
}

func (b *Bounds) BndToTrs(o *Bounds) (trs *mat.Dense) {
	r, c := b.mins.Dims()
	aSize := c
	trs = mat.NewDense(r, aSize*2, nil)
	move := trs.Slice(0, r, 0, aSize).(*mat.Dense)
	scale := trs.Slice(0, r, c, aSize*2).(*mat.Dense)

	move.Sub(o.Centers(), b.Centers())
	scale.Apply(func(i, j int, v float64) float64 {
		bSub := b.Subs().At(i, j)
		oSub := o.Subs().At(i, j)
		if bSub == 0 || oSub == 0 {
			return 0
		}
		return math.Log(oSub / bSub)
	}, scale)
	move.Apply(func(i, j int, v float64) float64 {
		bSub := b.Subs().At(i, j)
		if bSub == 0 {
			return 0
		}
		return v / bSub
	}, move)
	return
}

func (b *Bounds) TrsToBnd(trs *mat.Dense) (o *Bounds) {
	r, c := trs.Dims()
	aSize := c / 2
	o = NewBoxs(r, aSize)
	move := trs.Slice(0, r, 0, aSize).(*mat.Dense)
	scale := trs.Slice(0, r, aSize, aSize*2).(*mat.Dense)

	o.ctrs.MulElem(move, b.Subs())
	o.subs.Apply(func(i, j int, v float64) float64 { return math.Exp(v) }, scale)

	o.ctrs.Add(o.ctrs, b.Centers())
	o.subs.MulElem(o.subs, b.Subs())

	o.SetByBox()
	return
}

func (b *Bounds) ToDense() *mat.Dense {
	cnt, size := b.Dims()
	ret := mat.NewDense(cnt, size*2, nil)
	b.SetToDense(ret)
	return ret
}

func (b *Bounds) SetToDense(ret *mat.Dense) {
	cnt, size := b.Dims()
	ret.Slice(0, cnt, 0, size).(*mat.Dense).Copy(b.mins)
	ret.Slice(0, cnt, size, size*2).(*mat.Dense).Copy(b.maxs)
}
