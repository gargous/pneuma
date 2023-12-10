package nn

import (
	"math"
	"pneuma/common"

	"gonum.org/v1/gonum/mat"
)

type LossParam struct {
	Threshold float64
	MinLoss   float64
	MinTimes  int
}

func NewLossParam() *LossParam {
	return &LossParam{
		Threshold: 0.01,
		MinLoss:   0.01,
		MinTimes:  1000,
	}
}

func (l *LossParam) Copy(src *LossParam) {
	l.Threshold = src.Threshold
	l.MinLoss = src.MinLoss
	l.MinTimes = src.MinTimes
}

type loss struct {
	losses []float64
	target common.ITarget
	param  *LossParam
}

func (l *loss) copy(src *loss) {
	l.param = &LossParam{}
	*l.param = *src.param
	copy(l.losses, src.losses)
	l.target = common.CopyITarget(src.target)
}

func (l *loss) check(pred, targ *mat.Dense) bool {
	loss := l.target.Loss(pred, targ)
	l.losses = append(l.losses, loss)
	return l.isDone()
}

func (l *loss) backward() *mat.Dense {
	return l.target.Backward()
}

func (l *loss) isDone() bool {
	param := l.param
	if param == nil {
		return false
	}
	if len(l.losses) <= param.MinTimes {
		return false
	}
	lossCur := l.losses[len(l.losses)-1]
	lossOld := l.losses[len(l.losses)-2]
	if math.Abs(lossCur-lossOld) > param.Threshold {
		return false
	}
	return lossCur <= param.MinLoss
}

type TargetMSE struct {
	sub *mat.Dense
}

func NewTarMSE() *TargetMSE {
	return &TargetMSE{}
}

func (t *TargetMSE) Copy(src *TargetMSE) {
	if src.sub != nil {
		t.sub = &mat.Dense{}
		t.sub.CloneFrom(src.sub)
	}
}

func (t *TargetMSE) Loss(pred, targ *mat.Dense) (y float64) {
	r, c := pred.Dims()
	sub := mat.NewDense(r, c, nil)
	sub.Sub(pred, targ)
	cnt := float64(r*c) * 2
	for j := 0; j < c; j++ {
		col := sub.ColView(j)
		y += mat.Dot(col, col)
	}
	t.sub = sub
	y /= cnt
	return
}

func (t *TargetMSE) Backward() (dy *mat.Dense) {
	return t.sub
}

type TargetCE struct {
	softmax *mat.Dense
	target  *mat.Dense
}

func NewTarCE() *TargetMSE {
	return &TargetMSE{}
}

func (t *TargetCE) Copy(src *TargetCE) {
	if src.softmax != nil {
		t.softmax = &mat.Dense{}
		t.softmax.CloneFrom(src.softmax)
		t.target = &mat.Dense{}
		t.target.CloneFrom(src.target)
	}
}

func (t *TargetCE) Loss(pred, targ *mat.Dense) (y float64) {
	r, c := pred.Dims()
	softmax := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		col := pred.ColView(j)
		max := mat.Max(col)
		sumExp := 0.0
		softmaxCol := mat.NewVecDense(r, nil)
		for i := 0; i < r; i++ {
			exp := math.Exp(col.AtVec(i) - max)
			sumExp += exp
			softmaxCol.SetVec(i, exp)
		}
		softmaxCol.ScaleVec(1.0/sumExp, softmaxCol)
		softmax.SetCol(j, softmaxCol.RawVector().Data)
		logSumExp := math.Log(sumExp)
		for i := 0; i < r; i++ {
			y -= targ.At(i, j) * (pred.At(i, j) - max - logSumExp)
		}
	}
	t.softmax = softmax
	t.target = targ
	y /= float64(c)
	return
}

func (t *TargetCE) Backward() (dy *mat.Dense) {
	r, c := t.target.Dims()
	dy = mat.NewDense(r, c, nil)
	dy.Sub(t.softmax, t.target)
	return
}
