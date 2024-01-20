package nn

import (
	"math"
	"pneuma/common"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type LossParam struct {
	Threshold float64
	MinLoss   float64
	MinTimes  int
}

func NewLossParam() *LossParam {
	return &LossParam{
		Threshold: 0.0001,
		MinLoss:   0.0001,
		MinTimes:  1000,
	}
}

func (param *LossParam) IsDone(losses []float64) bool {
	if param == nil {
		return false
	}
	if len(losses) <= param.MinTimes {
		return false
	}
	lossCur := losses[len(losses)-1]
	lossOld := losses[len(losses)-2]
	if math.Abs(lossCur-lossOld) > param.Threshold {
		return false
	}
	return lossCur <= param.MinLoss
}

type loss struct {
	losses []float64
	target common.ITarget
	param  *LossParam
}

func (l *loss) forward(pred, targ *mat.Dense) {
	loss := l.target.Loss(pred, targ)
	l.losses = append(l.losses, loss)
}

func (l *loss) backward() *mat.Dense {
	return l.target.Backward()
}

func (l *loss) isDone() bool {
	return l.param.IsDone(l.losses)
}

// cross entropy
type TargetCE struct {
	softmax *mat.Dense
	target  *mat.Dense
}

func NewTarCE() *TargetCE {
	return &TargetCE{}
}

func (t *TargetCE) LossEach(pred, targ *mat.Dense) (loss *mat.Dense) {
	r, c := pred.Dims()
	softmax := mat.NewDense(r, c, nil)
	loss = mat.NewDense(r, c, nil)
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
			loss.Set(i, j, -1*targ.At(i, j)*(pred.At(i, j)-max-logSumExp))
		}
	}
	t.softmax = softmax
	t.target = targ
	return
}

func (t *TargetCE) Loss(pred, targ *mat.Dense) (y float64) {
	_, batch := pred.Dims()
	return mat.Sum(t.LossEach(pred, targ)) / float64(batch)
}

func (t *TargetCE) Backward() (dy *mat.Dense) {
	r, c := t.target.Dims()
	dy = mat.NewDense(r, c, nil)
	dy.Sub(t.softmax, t.target)
	return
}

func (t *TargetCE) Acc(pred, targ *mat.Dense) (acc float64) {
	cnt := 0.0
	_, batch := targ.Dims()
	for j := 0; j < batch; j++ {
		pred := mat.Col(nil, j, pred)
		targ := mat.Col(nil, j, targ)
		maxPredIdx := floats.MaxIdx(pred)
		maxTargIdx := floats.MaxIdx(targ)
		if maxPredIdx == maxTargIdx {
			cnt++
		}
	}
	return cnt / float64(batch)
}

func LossToAccLinear(loss mat.Matrix) (acc float64) {
	_, batch := loss.Dims()
	cnt := 0.0
	for j := 0; j < batch; j++ {
		oneLoss := mat.Col(nil, j, loss)
		min := floats.Min(oneLoss)
		dis := floats.Max(oneLoss) - min
		ret := 0.0
		for i := 0; i < len(oneLoss); i++ {
			ret += 1 - (oneLoss[i]-min)/dis
		}
		cnt += ret / float64(len(oneLoss))
	}
	return cnt / float64(batch)
}

// l1
type TargetMAE struct {
	dy *mat.Dense
}

func NewTarMAE() *TargetMAE {
	return &TargetMAE{}
}

func (t *TargetMAE) LossEach(pred, targ *mat.Dense) (loss *mat.Dense) {
	r, c := pred.Dims()
	loss = mat.NewDense(r, c, nil)
	loss.Sub(pred, targ)
	loss.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, loss)
	return loss
}

func (t *TargetMAE) Loss(pred, targ *mat.Dense) (y float64) {
	r, c := pred.Dims()
	cnt := float64(r * c)
	loss := t.LossEach(pred, targ)
	t.dy = mat.NewDense(r, c, nil)
	t.dy.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return -1
	}, loss)
	y = mat.Sum(loss) / cnt
	return
}

func (t *TargetMAE) Backward() (dy *mat.Dense) {
	return t.dy
}

func (t *TargetMAE) Acc(pred, targ *mat.Dense) (acc float64) {
	return LossToAccLinear(t.LossEach(pred, targ))
}

// smooth l1
type TargetSmoothMAE struct {
	beta float64
	dy   *mat.Dense
}

func NewTarSmoothMAE(beta float64) *TargetSmoothMAE {
	return &TargetSmoothMAE{beta: beta}
}

func (t *TargetSmoothMAE) LossEach(pred, targ *mat.Dense) (loss *mat.Dense) {
	r, c := pred.Dims()
	t.dy = mat.NewDense(r, c, nil)
	loss = mat.NewDense(r, c, nil)
	loss.Sub(pred, targ)
	loss.Apply(func(i, j int, v float64) float64 {
		a := math.Abs(v)
		if a < t.beta {
			t.dy.Set(i, j, v)
			return 0.5 * a * a / t.beta
		}
		if v > 0 {
			t.dy.Set(i, j, 1)
		} else {
			t.dy.Set(i, j, -1)
		}
		return a - 0.5*t.beta
	}, loss)
	return
}

func (t *TargetSmoothMAE) Acc(pred, targ *mat.Dense) (acc float64) {
	loss := t.LossEach(pred, targ)
	return LossToAccLinear(loss)
}

func (t *TargetSmoothMAE) Loss(pred, targ *mat.Dense) (y float64) {
	r, c := pred.Dims()
	cnt := float64(r * c)
	loss := t.LossEach(pred, targ)
	y = mat.Sum(loss) / cnt
	return
}

func (t *TargetSmoothMAE) Backward() (dy *mat.Dense) {
	return t.dy
}

// l2
type TargetMSE struct {
	sub *mat.Dense
}

func NewTarMSE() *TargetMSE {
	return &TargetMSE{}
}

func (t *TargetMSE) Loss(pred, targ *mat.Dense) (y float64) {
	r, c := pred.Dims()
	cnt := float64(r * c)
	loss := t.LossEach(pred, targ)
	y = mat.Sum(loss) / cnt
	return
}

func (t *TargetMSE) Acc(pred, targ *mat.Dense) (acc float64) {
	loss := t.LossEach(pred, targ)
	return LossToAccLinear(loss)
}

func (t *TargetMSE) Backward() (dy *mat.Dense) {
	return t.sub
}

func (t *TargetMSE) LossEach(pred, targ *mat.Dense) (loss *mat.Dense) {
	r, c := pred.Dims()
	t.sub = mat.NewDense(r, c, nil)
	t.sub.Sub(pred, targ)
	loss = mat.NewDense(r, c, nil)
	loss.Apply(func(i, j int, v float64) float64 {
		return v * v * 0.5
	}, t.sub)
	return
}
