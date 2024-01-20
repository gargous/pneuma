package frcnn

import (
	"math"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/mat"
)

type RPNLossParam struct {
	*nn.LossParam
	ScoreAlpha float64
}

func NewRPNLossParam() *RPNLossParam {
	return &RPNLossParam{
		LossParam:  nn.NewLossParam(),
		ScoreAlpha: 0.5,
	}
}

type rpnLoss struct {
	losses []float64
	score  *nn.TargetCE
	trans  common.ITarget
	bnds   common.ITarget
	param  *RPNLossParam
}

func newRPNTrsLoss(tar common.ITarget, param *RPNLossParam) *rpnLoss {
	return &rpnLoss{
		trans: tar,
		score: nn.NewTarCE(),
		param: param,
	}
}

func newRPNBndLoss(tar common.ITarget, param *RPNLossParam) *rpnLoss {
	return &rpnLoss{
		bnds:  tar,
		score: nn.NewTarCE(),
		param: param,
	}
}

func (l *rpnLoss) scoresFlat(scores *mat.Dense) (newScores *mat.Dense) {
	sCnt, batch := scores.Dims()
	pCnt := sCnt / 2
	newScores = mat.NewDense(2, batch*pCnt, nil)
	for i := 0; i < pCnt; i++ {
		dstIdx := i * batch
		dstSlice := newScores.Slice(0, 2, dstIdx, dstIdx+batch).(*mat.Dense)
		srcIdx := i * 2
		srcSlice := scores.Slice(srcIdx, srcIdx+2, 0, batch)
		dstSlice.Copy(srcSlice)
	}
	return
}

func (l *rpnLoss) scoresRestore(base *Bounds, scores *mat.Dense) (newScores *mat.Dense) {
	_, sAll := scores.Dims()
	pCnt := base.Len()
	batch := sAll / pCnt
	newScores = mat.NewDense(pCnt*2, batch, nil)
	for i := 0; i < pCnt; i++ {
		srcIdx := i * batch
		srcSlice := scores.Slice(0, 2, srcIdx, srcIdx+batch)
		dstIdx := i * 2
		dstSlice := newScores.Slice(dstIdx, dstIdx+2, 0, batch).(*mat.Dense)
		dstSlice.Copy(srcSlice)
	}
	return
}

func (l *rpnLoss) mergeTestRet(lables, scoreLoss, transLoss *mat.Dense) (loss, acc float64) {
	pCnt, batch := lables.Dims()
	sCnt, _ := scoreLoss.Dims()
	tCnt, _ := transLoss.Dims()
	sSize := sCnt / pCnt
	tSize := tCnt / pCnt
	lossCnt := 0.0
	accCnt := 0.0
	common.RecuRange([]int{pCnt, batch}, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		sIdx := i * sSize
		tIdx := i * tSize
		lab := lables.At(i, j)
		if lab == RPNLabIgn {
			return
		}
		score := scoreLoss.Slice(sIdx, sIdx+sSize, j, j+1)
		trans := transLoss.Slice(tIdx, tIdx+tSize, j, j+1)
		loss += mat.Sum(score)*l.param.ScoreAlpha/float64(sSize) + mat.Sum(trans)*(1-l.param.ScoreAlpha)/float64(tSize)
		lossCnt++
		if lab == RPNLabPos {
			acc += nn.LossToAccLinear(trans)
			accCnt++
		}
	})
	loss /= lossCnt
	acc /= accCnt
	if lossCnt == 0 {
		loss = 0
	}
	if accCnt == 0 {
		acc = 0
	}
	return
}

func (l *rpnLoss) test(base *Bounds, lables, scoresPD, scoresTG, transfPD *mat.Dense, bndsTG []*Bounds) (loss, acc float64) {
	var scoreLoss, transLoss *mat.Dense
	scoreLoss = l.scoresRestore(base, l.score.LossEach(l.scoresFlat(scoresPD), l.scoresFlat(scoresTG)))
	batch := len(bndsTG)
	r, c := base.Dims()
	switch {
	case l.trans != nil:
		tgDense := mat.NewDense(r*c*2, batch, nil)
		for j := 0; j < batch; j++ {
			transf := base.BndToTrs(bndsTG[j])
			tgDense.SetCol(j, transf.RawMatrix().Data)
		}
		pdDense := transfPD
		transLoss = l.trans.LossEach(pdDense, tgDense)
	case l.bnds != nil:
		pdDense := mat.NewDense(r*c*2, batch, nil)
		tgDense := mat.NewDense(r*c*2, batch, nil)
		eyeBnd := NewEyeBounds(r, c)
		for j := 0; j < batch; j++ {
			transfData := mat.Col(nil, j, transfPD)
			transf := mat.NewDense(r, c*2, transfData)
			slicePD := pdDense.Slice(0, r*c*2, j, j+1).(*mat.Dense)
			sliceTG := tgDense.Slice(0, r*c*2, j, j+1).(*mat.Dense)
			eyeBnd.TrsToBnd(transf).SetToDense(slicePD)
			bndsTG[j].SetToDense(sliceTG)
		}
		transLoss = l.bnds.LossEach(pdDense, tgDense)
	}
	return l.mergeTestRet(lables, scoreLoss, transLoss)
}

func (l *rpnLoss) forward(base *Bounds, lables, scoresPD, scoresTG, transfPD *mat.Dense, bndsTG []*Bounds) {
	loss, _ := l.test(base, lables, scoresPD, scoresTG, transfPD, bndsTG)
	l.losses = append(l.losses, loss)
}

func (l *rpnLoss) backward(base *Bounds) (dScores, dTransf *mat.Dense) {
	dScores = l.scoresRestore(base, l.score.Backward())
	dScores.Scale(l.param.ScoreAlpha, dScores)
	switch {
	case l.trans != nil:
		dTransf = l.trans.Backward()
	case l.bnds != nil:
		dTransf = l.bnds.Backward()
	}
	dTransf.Scale(1-l.param.ScoreAlpha, dTransf)
	return
}

func (l *rpnLoss) isDone() bool {
	return l.param.IsDone(l.losses)
}

type TargetIOU struct {
	r        int
	c        int
	ious     []*mat.VecDense
	selects  []*mat.Dense
	predBnds []*Bounds
}

func NewTarIOU(r, c int) *TargetIOU {
	return &TargetIOU{r: r, c: c}
}

func (t *TargetIOU) Loss(pred, targ *mat.Dense) (y float64) {
	r, batch := pred.Dims()
	t.predBnds = make([]*Bounds, batch)
	t.ious = make([]*mat.VecDense, batch)
	t.selects = make([]*mat.Dense, batch)
	colData := make([]float64, r)
	for j := 0; j < batch; j++ {
		predColMat := mat.NewDense(t.r, t.c*2, mat.Col(colData, j, pred))
		targColMat := mat.NewDense(t.r, t.c*2, mat.Col(colData, j, targ))
		bndPred := NewBounds(t.r, t.c)
		bndPred.SetAll(predColMat)
		bndTarg := NewBounds(t.r, t.c)
		bndTarg.SetAll(targColMat)
		ious, selects := bndPred.IOUElem(bndTarg)
		for i := 0; i < t.r; i++ {
			y -= math.Log(ious.AtVec(i))
		}
		t.predBnds[j] = bndPred
		t.ious[j] = ious
		t.selects[j] = selects
	}
	y /= float64(batch * t.r)
	return
}

func (t *TargetIOU) Backward() (dy *mat.Dense) {
	batch := len(t.selects)
	dy = mat.NewDense(t.r*t.c*2, batch, nil)
	colMat := mat.NewDense(t.r, t.c*2, nil)
	colIOUMat := mat.NewDense(t.r, t.c*2, nil)
	for j := 0; j < batch; j++ {
		subs := t.predBnds[j].Subs()
		ctrs := t.predBnds[j].Centers()
		sele := t.selects[j]
		ious := t.ious[j]
		for i := 0; i < t.c*2; i++ {
			colIOUMat.SetCol(i, ious.RawVector().Data)
		}
		colMat.Slice(0, t.r, 0, t.c).(*mat.Dense).Copy(subs)
		colMat.Slice(0, t.r, t.c, t.c*2).(*mat.Dense).Copy(ctrs)
		colMat.Sub(colMat, sele)
		colMat.MulElem(colMat, colIOUMat)
		colMat.Sub(colMat, sele)
		dy.SetCol(j, colMat.RawMatrix().Data)
	}
	return
}
