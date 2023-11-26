package cnn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type hLayerConvCore struct {
	w  *mat.Dense
	b  *mat.Dense
	dw *mat.Dense
	db *mat.Dense
}

type HLayerConv2D struct {
	size    []int
	stride  []int
	padding bool
	cores   []*hLayerConvCore
	x       []*mat.Dense
}

func (l *HLayerConv2D) newHLayerConvCore(coreSize []int) *hLayerConvCore {
	coreCnt := l.coreCnt(coreSize)
	br := coreCnt[0]
	bc := coreCnt[1]
	wr := coreSize[0]
	wc := coreSize[1]
	ret := &hLayerConvCore{}
	ret.w = mat.NewDense(wr, wc, nil)
	ret.b = mat.NewDense(br, bc, nil)
	for i := 0; i < wr*wc; i++ {
		ret.w.RawMatrix().Data[i] = rand.Float64()
	}
	for i := 0; i < br*bc; i++ {
		ret.b.RawMatrix().Data[i] = rand.Float64()
	}
	return ret
}

func (l *HLayerConv2D) coreCnt(coreSize []int) (cnts []int) {
	cnts = make([]int, len(l.size))
	for i := 0; i < len(cnts); i++ {
		cnts[i] = 1 + (l.size[i]-coreSize[i])/l.stride[i]
		if (l.size[i]-coreSize[i])%l.stride[i] != 0 && l.padding {
			cnts[i] += 1
		}
	}
	return
}

func (l *HLayerConv2D) paddingCnt(coreSize []int) (cnts []int) {
	cnts = make([]int, len(l.size))
	for i := 0; i < len(cnts); i++ {
		cnt := (l.size[i] - coreSize[i]) % l.stride[i]
		if cnt != 0 && l.padding {
			cnts[i] = cnt
		}
	}
	return
}

func (l *HLayerConv2D) parse2D(data *mat.Dense) []*mat.Dense {
	_, c := data.Dims()
	ret := make([]*mat.Dense, c)
	corer, corec := l.cores[0].w.Dims()
	paddings := l.paddingCnt([]int{corer, corec})
	for j := 0; j < c; j++ {
		col := mat.Col(nil, j, data)
		onePlain := mat.NewDense(l.size[0], l.size[1], col)
		if paddings[0] > 0 && paddings[1] > 0 {
			rr := l.size[0] + paddings[0]
			rc := l.size[1] + paddings[1]
			il := paddings[0] / 2
			jt := paddings[1] / 2
			tmpPlain := mat.NewDense(rr, rc, nil)
			tmpPlain.Slice(il, il+l.size[0], jt, jt+l.size[1]).(*mat.Dense).Copy(onePlain)
			onePlain = tmpPlain
		}
		ret[j] = onePlain
	}
	return ret
}

func (l *HLayerConv2D) Forward(x *mat.Dense) (y *mat.Dense) {
	l.x = l.parse2D(x)
	return
}

func (l *HLayerConv2D) Backward(dy *mat.Dense) (dx *mat.Dense) {
	return
}
