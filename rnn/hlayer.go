package rnn

import (
	"math/rand"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

type HLayerRNN struct {
	outLay   *nn.HLayerLinear
	u        *mat.Dense
	w        *mat.Dense
	b        *mat.VecDense
	x        []*mat.Dense
	s        []*mat.Dense
	sPred    *mat.Dense
	du       []*mat.Dense
	dw       []*mat.Dense
	db       []*mat.VecDense
	diagSSum *mat.Dense
	act      common.IHLayer
	seqSize  int
	seqIdx   int
}

func NewHLayerCommonRNN(seqSize int) *HLayerRNN {
	return NewHLayerRNN(seqSize, nn.NewHLayerTanh())
}

func NewHLayerRNN(seqSize int, act common.IHLayer) *HLayerRNN {
	return &HLayerRNN{
		outLay:  nn.NewHLayerLinear(),
		act:     act,
		seqSize: seqSize,
	}
}

func (l *HLayerRNN) InitSize(size []int) []int {
	r, c := size[0], size[1]
	l.w = mat.NewDense(l.seqSize, l.seqSize, nil)
	l.u = mat.NewDense(l.seqSize, r, nil)
	l.b = mat.NewVecDense(l.seqSize, nil)
	l.outLay.InitSize([]int{c, l.seqSize})
	l.w.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, l.w)
	l.u.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, l.w)
	l.diagSSum = mat.NewDense(l.seqSize, l.seqSize, nil)
	return size
}

func (l *HLayerRNN) SeqReset() {
	l.dw = nil
	l.du = nil
	l.db = nil
	l.s = nil
	l.seqIdx = 0
	l.sPred = nil
}

func (l *HLayerRNN) Predict(x *mat.Dense) (y *mat.Dense) {
	_, batch := x.Dims()
	s := mat.NewDense(l.seqSize, batch, nil)
	s.Mul(l.u, x)
	if l.sPred != nil {
		newS := mat.NewDense(l.seqSize, batch, nil)
		newS.Mul(l.w, l.sPred)
		s.Add(s, newS)
	}
	s = l.act.Forward(s)
	y = l.outLay.Forward(s)
	l.sPred = s
	return
}

func (l *HLayerRNN) Forward(x *mat.Dense) (y *mat.Dense) {
	_, batch := x.Dims()
	s := mat.NewDense(l.seqSize, batch, nil)
	s.Mul(l.u, x)
	if l.s != nil {
		oldS := l.s[len(l.s)-1]
		newS := mat.NewDense(l.seqSize, batch, nil)
		newS.Mul(l.w, oldS)
		s.Add(s, newS)
	}
	s = l.act.Forward(s)
	l.s = append(l.s, s)
	l.x = append(l.x, x)
	y = l.outLay.Forward(s)
	return
}

func (l *HLayerRNN) Backward(dy *mat.Dense) (dx *mat.Dense) {
	_, batch := dy.Dims()
	ds := l.act.Backward(l.outLay.Backward(dy))
	ur, uc := l.u.Dims()
	wr, wc := l.w.Dims()
	du := mat.NewDense(ur, uc, nil)
	dw := mat.NewDense(wr, wc, nil)
	db := mat.NewVecDense(l.b.Len(), nil)
	du.Mul(ds, l.x[l.seqIdx].T())
	dw.Mul(ds, l.s[l.seqIdx].T())
	for j := 0; j < batch; j++ {
		db.AddVec(db, ds.ColView(j))
	}
	if l.seqIdx > 0 {
		newDu := mat.NewDense(ur, uc, nil)
		newDw := mat.NewDense(wr, wc, nil)
		newDb := mat.NewVecDense(l.b.Len(), nil)
		for i := 0; i < l.b.Len(); i++ {
			l.diagSSum.Set(i, i, l.b.AtVec(i))
		}
		wt := l.w.T()
		newDu.Mul(wt, l.du[l.seqIdx-1])
		newDu.Mul(l.diagSSum, newDu)
		du.Add(du, newDu)
		newDw.Mul(wt, l.dw[l.seqIdx-1])
		newDw.Mul(l.diagSSum, newDw)
		dw.Add(dw, newDw)
		newDb.MulVec(wt, l.db[l.seqIdx-1])
		newDb.MulVec(l.diagSSum, newDb)
		db.AddVec(db, newDb)
	}
	l.du = append(l.du, du)
	l.dw = append(l.dw, dw)
	l.db = append(l.db, db)
	l.seqIdx++
	return
}

func (l *HLayerRNN) Optimize() (datas, deltas []mat.Matrix) {
	datas = []mat.Matrix{
		l.w, l.u, l.b,
	}
	deltas = []mat.Matrix{
		SumDense(l.dw), SumDense(l.du), SumVecDense(l.db),
	}
	return
}

type HLayerLSTM struct {
}
