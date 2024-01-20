package frcnn

import (
	"fmt"
	"math"
	"math/rand"
	"pneuma/cnn"
	"pneuma/common"
	"pneuma/nn"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	RPNLabPos, RPNLabNeg, RPNLabIgn = 1.0, 2.0, 3.0
)

var (
	RPNLabVecs   = common.OneHotVecs(2)
	RPNLabPosVec = RPNLabVecs[0]
	RPNLabNegVec = RPNLabVecs[1]
)

type RPNParam struct {
	AnchorGroup        int
	AnchorRatio        float64
	OrgSize, RoiSize   []int
	NegIOU, PosIOU     float64
	NegRatio, PosRatio float64
}

func NewRPNParam(orgSize, roiSize []int) *RPNParam {
	return &RPNParam{
		AnchorGroup: 3,
		AnchorRatio: 2,
		OrgSize:     orgSize,
		RoiSize:     roiSize,
		NegIOU:      0.3,
		PosIOU:      0.7,
		NegRatio:    0.5,
		PosRatio:    0.5,
	}
}

type RPN struct {
	param      *RPNParam
	loss       *rpnLoss
	anchorPerG int
	anchors    *mat.Dense
	convScores common.IHLayerSizeIniter
	convTransf common.IHLayerSizeIniter
	PropInners []bool
	PropBounds *Bounds
	opt        common.IOptimizer
}

func NewRPN(param *RPNParam) *RPN {
	anchorPerG := (1 + 2*int(param.AnchorRatio-0.5))
	ret := &RPN{
		param:      param,
		anchorPerG: anchorPerG,
		anchors:    mat.NewDense(param.AnchorGroup*anchorPerG, len(param.OrgSize)-1, nil),
	}
	ret.SetTrsTarget(nn.NewTarSmoothMAE(1), NewRPNLossParam())
	return ret
}

func (l *RPN) SetOpt(opt common.IOptimizer) {
	l.opt = opt
}

func (l *RPN) SetTrsTarget(tar common.ITarget, param *RPNLossParam) {
	l.loss = newRPNTrsLoss(tar, param)
}

func (l *RPN) SetBndTarget(tar common.ITarget, param *RPNLossParam) {
	l.loss = newRPNBndLoss(tar, param)
}

func (l *RPN) ScoreLayerParam() cnn.ConvKernalParam {
	aCnt, _ := l.anchors.Dims()
	dlen := len(l.param.OrgSize) - 1
	dstri := common.IntsAddConst(1, make([]int, dlen))
	size := append(common.IntsAddConst(1, make([]int, dlen)), aCnt*2)
	return cnn.NewConvKParam(size, dstri, cnn.ConvKernalPadFit)
}

func (l *RPN) TransLayerParam() cnn.ConvKernalParam {
	aCnt, aSize := l.anchors.Dims()
	dlen := len(l.param.OrgSize) - 1
	dstri := common.IntsAddConst(1, make([]int, dlen))
	size := append(common.IntsAddConst(1, make([]int, dlen)), aCnt*aSize*2)
	return cnn.NewConvKParam(size, dstri, cnn.ConvKernalPadFit)
}

func (l *RPN) SetScoreLayer(lay common.IHLayerSizeIniter) {
	l.convScores = lay
}

func (l *RPN) SetTransLayer(lay common.IHLayerSizeIniter) {
	l.convTransf = lay
}

func (l *RPN) InitSize(size []int) []int {
	if l.convScores == nil {
		l.SetScoreLayer(cnn.NewHLayerConv(l.ScoreLayerParam()))
		l.SetTransLayer(cnn.NewHLayerConv(l.TransLayerParam()))
	}
	l.convScores.InitSize(size)
	minSize := l.convTransf.InitSize(size)
	minSize = minSize[:len(minSize)-1]
	orgSize := l.param.OrgSize[:len(l.param.OrgSize)-1]
	l.genAnchor(orgSize, minSize)
	l.genPropBound(orgSize, minSize)
	return l.param.RoiSize
}

func (l *RPN) genAnchor(orgSize, minSize []int) {
	base := common.IntsDiv(orgSize, minSize)
	param := l.param
	perG := l.anchorPerG
	idx := 0
	rates := make([]float64, perG)
	for i := -perG / 2; i <= perG/2; i++ {
		rate := math.Sqrt(math.Pow(param.AnchorRatio, float64(i)))
		rates[idx] = rate
		idx += 1
	}
	idx = 0
	scales := make([]float64, l.param.AnchorGroup)
	scaleStart := float64(l.param.AnchorGroup-1) / 2
	for i := 0; i < l.param.AnchorGroup; i++ {
		scale := math.Pow(2, scaleStart)
		scales[idx] = scale
		idx += 1
		scaleStart += 1
	}
	idx = 0
	for _, rate := range rates {
		for _, scale := range scales {
			h := float64(base[0]) * rate * scale
			w := float64(base[1]) / rate * scale
			anchors := l.anchors.RowView(idx).(*mat.VecDense)
			anchors.SetVec(0, h/2)
			anchors.SetVec(1, w/2)
			idx += 1
		}
	}
}

func (l *RPN) genPropBound(orgSize, minSize []int) {
	aCnt, aSize := l.anchors.Dims()
	orgBoundData := common.IntsToF64s(orgSize)
	orgCenterPos := make([]float64, len(minSize))
	baseCenterPos := common.IntsToF64s(minSize)

	eachCenter := mat.NewDense(aCnt, aSize, nil)
	eachPos := make([]float64, len(minSize))
	offset := make([]float64, len(minSize))
	stride := common.IntsDiv(orgSize, minSize)

	floats.DivTo(offset, orgBoundData, baseCenterPos)
	floats.Scale(0.5, offset)
	floats.ScaleTo(orgCenterPos, 0.5, orgBoundData)
	floats.Scale(0.5, baseCenterPos)

	pSize := aSize * 2
	pCnt := common.IntsProd(minSize) * aCnt
	idx := 0
	propBounds := mat.NewDense(pCnt, pSize, nil)
	l.PropInners = make([]bool, pCnt)

	common.RecuRange(orgSize, stride, func(pos []int) {
		common.IntsToF64sTo(eachPos, pos)
		floats.Add(eachPos, offset)
		for i := 0; i < aCnt; i++ {
			eachCenter.SetRow(i, eachPos)
		}
		sliceMin := propBounds.Slice(idx, idx+aCnt, 0, aSize).(*mat.Dense)
		sliceMax := propBounds.Slice(idx, idx+aCnt, aSize, pSize).(*mat.Dense)
		sliceMin.Sub(eachCenter, l.anchors)
		sliceMax.Add(eachCenter, l.anchors)
		idx += aCnt
	})

	propSubData := make([]float64, aSize)
	for i := 0; i < pCnt; i++ {
		propRowData := propBounds.RawRowView(i)
		propMin := propRowData[:aSize]
		if floats.Min(propMin) < 0 {
			continue
		}
		propMax := propRowData[aSize:pSize]
		floats.SubTo(propSubData, orgBoundData, propMax)
		if floats.Min(propSubData) < 0 {
			continue
		}
		l.PropInners[i] = true
	}
	l.PropBounds = NewBounds(pCnt, aSize)
	l.PropBounds.SetAll(propBounds)
}

func (l *RPN) forward(x *mat.Dense) (scores, transf *mat.Dense) {
	scores = l.convScores.Forward(x)
	transf = l.convTransf.Forward(x)
	return
}

func (l *RPN) predict(x *mat.Dense) (scores, transf *mat.Dense) {
	scores = common.Predic(l.convScores, x)
	transf = common.Predic(l.convTransf, x)
	return
}

func (l *RPN) genLables(maxIOUBIdxesOfP, maxIOUPOfB *mat.Dense) (lables *mat.Dense) {
	param := l.param
	pCnt, _ := maxIOUPOfB.Dims()
	bCnt, batch := maxIOUBIdxesOfP.Dims()

	lables = mat.NewDense(pCnt, batch, nil)
	posIdxs := mat.NewDense(pCnt, batch, nil)
	negIdxs := mat.NewDense(pCnt, batch, nil)
	posCnts := make([]int, batch)
	negCnts := make([]int, batch)
	common.RecuRange([]int{pCnt, batch}, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		if !l.PropInners[i] {
			lables.Set(i, j, RPNLabIgn)
			return
		}
		iou := maxIOUPOfB.At(i, j)
		if iou >= param.PosIOU {
			lables.Set(i, j, RPNLabPos)
		} else if iou < param.NegIOU {
			lables.Set(i, j, RPNLabNeg)
		} else {
			lables.Set(i, j, RPNLabIgn)
		}
	})
	common.RecuRange([]int{bCnt, batch}, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		idx := int(maxIOUBIdxesOfP.At(i, j))
		if !l.PropInners[idx] {
			return
		}
		lables.Set(idx, j, RPNLabPos)
	})
	common.RecuRange([]int{pCnt, batch}, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		v := lables.At(i, j)
		if v == RPNLabNeg {
			negIdxs.Set(negCnts[j], j, float64(i))
			negCnts[j]++
		} else if v == RPNLabPos {
			posIdxs.Set(posCnts[j], j, float64(i))
			posCnts[j]++
		}
	})
	posLimit := int(float64(pCnt) * param.PosRatio)
	negLimit := int(float64(pCnt) * param.NegRatio)
	for j := 0; j < batch; j++ {
		if posCnts[j] > posLimit {
			for _, idx := range rand.Perm(posCnts[j] - posLimit) {
				lables.Set(int(posIdxs.At(idx, j)), j, RPNLabIgn)
			}
		}
		if negCnts[j] > negLimit {
			for _, idx := range rand.Perm(negCnts[j] - negLimit) {
				lables.Set(int(negIdxs.At(idx, j)), j, RPNLabIgn)
			}
		}
	}
	return
}

func (l *RPN) genTarget(bound *mat.Dense) (lables, targScores *mat.Dense, targBounds []*Bounds) {
	br, batch := bound.Dims()
	_, aSize := l.anchors.Dims()
	pSize := aSize * 2
	pCnt := l.PropBounds.Len()
	bndMats := make([]*mat.Dense, batch)
	bCnt := br / pSize

	maxIOUBIdxesOfP := mat.NewDense(bCnt, batch, nil)
	maxIOUPIdxesOfB := mat.NewDense(pCnt, batch, nil)
	maxIOUPOfB := mat.NewDense(pCnt, batch, nil)
	iouRow := make([]float64, bCnt)
	iouCol := make([]float64, pCnt)
	for j := 0; j < batch; j++ {
		bndVecData := mat.Col(nil, j, bound)
		if len(bndVecData) != bCnt*pSize {
			fmt.Println(len(bndVecData), bCnt, pSize, aSize)
		}
		bndMats[j] = mat.NewDense(bCnt, pSize, bndVecData)
		bnds := NewBounds(bCnt, aSize)
		bnds.SetAll(bndMats[j])
		ious := l.PropBounds.IOUCross(bnds)
		for i := 0; i < bCnt; i++ {
			mat.Col(iouCol, i, ious)
			maxIOUBIdxesOfP.Set(i, j, float64(floats.MaxIdx(iouCol)))
		}
		for i := 0; i < pCnt; i++ {
			mat.Row(iouRow, i, ious)
			idx := floats.MaxIdx(iouRow)
			maxIOUPIdxesOfB.Set(i, j, float64(idx))
			maxIOUPOfB.Set(i, j, iouRow[idx])
		}
	}
	sSize := 2
	lables = l.genLables(maxIOUBIdxesOfP, maxIOUPOfB)
	targScores = mat.NewDense(pCnt*sSize, batch, nil)
	targBounds = make([]*Bounds, batch)
	common.RecuRange([]int{pCnt, batch}, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		sIdx := i * sSize
		labVal := lables.At(i, j)
		targBnds := targBounds[j]
		if targBnds == nil {
			targBnds = NewBounds(pCnt, aSize)
			targBounds[j] = targBnds
		}
		scoreSlice := targScores.Slice(sIdx, sIdx+sSize, j, j+1).(*mat.Dense)
		switch labVal {
		case RPNLabPos:
			scoreSlice.Copy(RPNLabPosVec)
			bIdx := int(maxIOUPIdxesOfB.At(i, j))
			bndVec := bndMats[j].RowView(bIdx).(*mat.VecDense)
			targBnds.SetRow(bIdx, bndVec)
		case RPNLabNeg:
			scoreSlice.Copy(RPNLabNegVec)
		}
	})
	return
}

func (l *RPN) backward(dScores, dTransf *mat.Dense) *mat.Dense {
	dXByScore := l.convScores.Backward(dScores)
	dXByTrans := l.convTransf.Backward(dTransf)
	dXByScore.Add(dXByScore, dXByTrans)
	return dXByScore
}

func (l *RPN) PredicTG(x *mat.Dense) (bound *mat.Dense) {
	//scoresPred, transfPred := l.predict(x)
	return
}

func (l *RPN) filterIgnoreLab(lables, data *mat.Dense) {
	dCnt, _ := data.Dims()
	pCnt, _ := lables.Dims()
	dSize := dCnt / pCnt
	data.Apply(func(i, j int, v float64) float64 {
		if lables.At(i/dSize, j) == RPNLabIgn {
			return 0
		}
		return v
	}, data)
}

func (l *RPN) Test(x, bound *mat.Dense) (loss, acc float64) {
	scoresPred, transfPred := l.predict(x)
	lables, scoresTarg, bnds := l.genTarget(bound)
	l.filterIgnoreLab(lables, scoresPred)
	l.filterIgnoreLab(lables, transfPred)
	loss, acc = l.loss.test(l.PropBounds, lables, scoresPred, scoresTarg, transfPred, bnds)
	return
}

func (l *RPN) Train(x, bound *mat.Dense) *mat.Dense {
	scoresPred, transfPred := l.forward(x)
	lables, scoresTarg, bnds := l.genTarget(bound)
	l.filterIgnoreLab(lables, scoresPred)
	l.filterIgnoreLab(lables, transfPred)
	l.loss.forward(l.PropBounds, lables, scoresPred, scoresTarg, transfPred, bnds)
	if l.loss.isDone() {
		return nil
	}
	dScores, dTransf := l.loss.backward(l.PropBounds)
	dx := l.backward(dScores, dTransf)
	l.opt.Update(common.OptimizeData(l.convScores, l.convTransf))
	return dx
}
