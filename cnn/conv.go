package cnn

import (
	"fmt"
	"pneuma/common"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

func paddingCnt(size, core, stride int, padding bool) (lp, rp, slip int) {
	if core > size {
		panic(fmt.Sprintf("paddingCnt need core bigger than size, now core=%d, size=%d", core, size))
	}
	slip = (size-core)/stride + 1
	nopadSize := (slip-1)*stride + core
	if nopadSize == size {
		return
	}
	p := 0
	if padding {
		p = nopadSize + stride - size
		slip += 1
	} else {
		p = nopadSize - size
	}
	lp = p / 2
	rp = p - lp
	return
}

type ConvKernalParam struct {
	size    []int
	stride  []int
	padding bool
}

func NewConvKParam(size, stride []int, padding bool) ConvKernalParam {
	return ConvKernalParam{
		size:    size,
		stride:  stride,
		padding: padding,
	}
}

type ConvPackerCalInfo struct {
	needFit     bool
	pads        []int
	fitPos      []int
	orgPos      []int
	step        int
	pad         int
	padOffset   int
	fitStride   []int
	kerOffset   int
	kerCnt      []int
	kerStartPos []int
	kerStride   []int
}

type ConvPacker struct {
	orgSize     []int
	orgSizeSum  int
	fitSize     []int
	fitSizeSum  int
	coreSize    []int
	stride      []int
	paddingLeft []int
	slipCnt     []int
	slipCntSum  int
	coreSizeSum int
	info        ConvPackerCalInfo
}

func NewConvPacker(inputSize []int, param ConvKernalParam) *ConvPacker {
	dim := len(inputSize)
	inpCnt := inputSize[len(inputSize)-1]
	ret := &ConvPacker{
		orgSize:     inputSize,
		coreSize:    append(param.size, inpCnt),
		stride:      append(param.stride, inpCnt),
		paddingLeft: make([]int, dim),
		fitSize:     make([]int, dim),
	}
	if len(ret.coreSize) != dim {
		panic(fmt.Sprintf("ConvPacker core size dims need equals to org:%d (%v) but %d (%v)", dim, inputSize, len(ret.coreSize), ret.coreSize))
	}
	if len(ret.stride) != dim {
		panic(fmt.Sprintf("ConvPacker stride dims need equals to org:%d (%v) but %d (%v)", dim, inputSize, len(ret.stride), ret.coreSize))
	}
	slips := make([]int, dim)
	for i := 0; i < dim; i++ {
		iinp := inputSize[i]
		icor := ret.coreSize[i]
		istr := ret.stride[i]
		pl, pr, slip := paddingCnt(iinp, icor, istr, param.padding)
		ret.paddingLeft[i] = pl
		slips[i] = slip
		ret.fitSize[i] = inputSize[i] + pl + pr
	}
	ret.slipCnt = slips
	ret.fitSizeSum = common.IntsProd(ret.fitSize)
	ret.orgSizeSum = common.IntsProd(ret.orgSize)
	ret.slipCntSum = common.IntsProd(slips)
	ret.coreSizeSum = common.IntsProd(ret.coreSize)

	step := dim - 2
	ret.info = ConvPackerCalInfo{
		needFit:     !common.IntsEqual(ret.orgSize, ret.fitSize),
		fitStride:   common.IntsAddConst(1, make([]int, dim)),
		kerStride:   common.IntsAddConst(1, make([]int, dim)),
		pads:        make([]int, dim),
		fitPos:      make([]int, dim),
		orgPos:      make([]int, dim),
		step:        step,
		pad:         ret.paddingLeft[step],
		padOffset:   common.IntsMin(ret.orgSize[step], ret.fitSize[step]) * inpCnt,
		kerOffset:   ret.coreSize[step] * inpCnt,
		kerCnt:      common.IntsAddConst(1, common.IntsSub(ret.fitSize, ret.coreSize)),
		kerStartPos: make([]int, dim),
	}
	copy(ret.info.pads[:step], ret.paddingLeft[:step])
	copy(ret.info.kerStride[step:], ret.coreSize[step:])
	copy(ret.info.fitStride[step:], ret.orgSize[step:])
	return ret
}

func (c *ConvPacker) SlipCnt() ([]int, int) {
	return c.slipCnt, c.slipCntSum
}

func (c *ConvPacker) CoreSize() ([]int, int) {
	return c.coreSize, c.coreSizeSum
}

func (c *ConvPacker) OrgSize() ([]int, int) {
	return c.orgSize, c.orgSizeSum
}

func sliceVec(v *mat.VecDense, from, to int) []float64 {
	data := v.RawVector().Data
	inc := v.RawVector().Inc
	return data[from*inc : (to-1)*inc+1]
}

func sliceVecCopy(dst, src *mat.VecDense, dstStart, srcStart, n int) {
	incDst := dst.RawVector().Inc
	incSrc := src.RawVector().Inc
	dstData := sliceVec(dst, dstStart, dstStart+n)
	srcData := sliceVec(src, srcStart, srcStart+n)
	imp := blas64.Implementation()
	imp.Dcopy(n, srcData, incSrc, dstData, incDst)
}

func sliceVecCopyToData(dst []float64, src *mat.VecDense, dstStart, srcStart, n int) {
	incSrc := src.RawVector().Inc
	dstData := dst[dstStart : dstStart+n]
	srcData := sliceVec(src, srcStart, srcStart+n)
	imp := blas64.Implementation()
	imp.Dcopy(n, srcData, incSrc, dstData, 1)
}

func sliceVecAdd(dst, src *mat.VecDense, dstStart, srcStart, n int) {
	incDst := dst.RawVector().Inc
	incSrc := src.RawVector().Inc
	dstData := sliceVec(dst, dstStart, dstStart+n)
	srcData := sliceVec(src, srcStart, srcStart+n)
	imp := blas64.Implementation()
	imp.Daxpy(n, 1, srcData, incSrc, dstData, incDst)
}

func (c *ConvPacker) Pack(vec *mat.VecDense) *mat.Dense {
	ret := mat.NewDense(c.slipCntSum, c.coreSizeSum, nil)
	c.PackTo(ret, vec)
	return ret
}

func (c *ConvPacker) UnPack(vec *mat.Dense) *mat.VecDense {
	ret := mat.NewVecDense(c.orgSizeSum, nil)
	c.UnPackTo(ret, vec)
	return ret
}

func (c *ConvPacker) PackTo(dst *mat.Dense, vec *mat.VecDense) {
	step := c.info.step
	fitVec := vec
	if c.info.needFit {
		fitVec = mat.NewVecDense(c.fitSizeSum, nil)
		common.RecuRange(c.orgSize, c.info.fitStride, func(orgPos []int) {
			common.IntsAddTo(c.info.fitPos, orgPos, c.info.pads)
			if common.SizeBound(c.info.fitPos, c.fitSize) {
				copy(c.info.orgPos, orgPos)
				if c.info.pad < 0 {
					c.info.orgPos[step] = -c.info.pad
				} else {
					c.info.fitPos[step] = c.info.pad
				}
				orgIdx := common.PosIdx(c.info.orgPos, c.orgSize)
				fitIdx := common.PosIdx(c.info.fitPos, c.fitSize)
				sliceVecCopy(fitVec, vec, fitIdx, orgIdx, c.info.padOffset)
			}
		})
	}

	slipRowIdx := 0
	common.RecuRange(c.info.kerCnt, c.stride, func(startPos []int) {
		slipRow := dst.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startOffset := startPos[step]
		copy(c.info.kerStartPos[:step], startPos[:step])
		common.RecuRange(c.coreSize, c.info.kerStride, func(corePos []int) {
			common.IntsAddTo(c.info.fitPos, corePos, c.info.kerStartPos)
			c.info.fitPos[step] = startOffset
			fitIdx := common.PosIdx(c.info.fitPos, c.fitSize)
			slipIdx := common.PosIdx(corePos, c.coreSize)
			sliceVecCopy(slipRow, fitVec, slipIdx, fitIdx, c.info.kerOffset)
		})
	})
}

func (c *ConvPacker) UnPackTo(dst *mat.VecDense, vec *mat.Dense) {
	step := c.info.step
	fitVec := dst
	if c.info.needFit {
		fitVec = mat.NewVecDense(c.fitSizeSum, nil)
	}
	slipRowIdx := 0
	common.RecuRange(c.info.kerCnt, c.stride, func(startPos []int) {
		slipRow := vec.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startOffset := startPos[step]
		copy(c.info.kerStartPos[:step], startPos[:step])
		common.RecuRange(c.coreSize, c.info.kerStride, func(corePos []int) {
			common.IntsAddTo(c.info.fitPos, corePos, c.info.kerStartPos)
			c.info.fitPos[step] = startOffset
			fitIdx := common.PosIdx(c.info.fitPos, c.fitSize)
			slipIdx := common.PosIdx(corePos, c.coreSize)
			sliceVecAdd(fitVec, slipRow, fitIdx, slipIdx, c.info.kerOffset)
		})
	})

	if c.info.needFit {
		common.RecuRange(c.orgSize, c.info.fitStride, func(orgPos []int) {
			common.IntsAddTo(c.info.fitPos, orgPos, c.info.pads)
			if common.SizeBound(c.info.fitPos, c.fitSize) {
				copy(c.info.orgPos, orgPos)
				if c.info.pad < 0 {
					c.info.orgPos[step] = -c.info.pad
				} else {
					c.info.fitPos[step] = c.info.pad
				}
				orgIdx := common.PosIdx(c.info.orgPos, c.orgSize)
				fitIdx := common.PosIdx(c.info.fitPos, c.fitSize)
				sliceVecCopy(dst, fitVec, orgIdx, fitIdx, c.info.padOffset)
			}
		})
	}
}

func (c *ConvPacker) FoldBatches(org, fld *mat.Dense, cb func(dst *mat.Dense, src *mat.VecDense)) {
	_, batch := org.Dims()
	fldRow, fldCol := fld.Dims()
	fldBatch := fldRow / batch
	for j := 0; j < batch; j++ {
		sliceFld := fld.Slice(j*fldBatch, j*fldBatch+fldBatch, 0, fldCol).(*mat.Dense)
		if cb == nil {
			sliceFldData := sliceFld.RawMatrix().Data
			orgCol := org.ColView(j).(*mat.VecDense)
			sliceVecCopyToData(sliceFldData, orgCol, 0, 0, len(sliceFldData))
		} else {
			cb(sliceFld, org.ColView(j).(*mat.VecDense))
		}
	}
}

func (c *ConvPacker) UnfoldBatches(org, fld *mat.Dense, cb func(dst *mat.VecDense, src *mat.Dense)) {
	_, batch := org.Dims()
	fldRow, fldCol := fld.Dims()
	fldBatch := fldRow / batch
	for j := 0; j < batch; j++ {
		sliceFld := fld.Slice(j*fldBatch, j*fldBatch+fldBatch, 0, fldCol).(*mat.Dense)
		if cb == nil {
			org.SetCol(j, sliceFld.RawMatrix().Data)
		} else {
			col := org.ColView(j).(*mat.VecDense)
			cb(col, sliceFld)
		}
	}
}

type MatColPicker struct {
	size    []int
	pickDim int
}

func NewMatColPicker(size []int, pickDim int) *MatColPicker {
	return &MatColPicker{
		size:    size,
		pickDim: pickDim,
	}
}

func (m *MatColPicker) PickTo(dst, data *mat.Dense) {
	_, c := data.Dims()
	searchStride := append([]int{}, m.size...)
	searchStride[m.pickDim] = 1
	rangStride := common.IntsAddConst(1, make([]int, len(m.size)))
	rangStride[m.pickDim] = m.size[m.pickDim]
	newColSize := append([]int{}, m.size...)
	newColSize[m.pickDim] = c
	pos := make([]int, len(m.size))
	newPos := make([]int, len(m.size))
	for i := 0; i < c; i++ {
		oldCol := data.ColView(i)
		common.RecuRange(m.size, searchStride, func(startPos []int) {
			newCol := dst.ColView(startPos[m.pickDim]).(*mat.VecDense)
			common.RecuRange(m.size, rangStride, func(rangPos []int) {
				common.IntsAddTo(pos, startPos, rangPos)
				oldIdx := common.PosIdx(pos, m.size)
				copy(newPos, pos)
				newPos[m.pickDim] = i
				newIdx := common.PosIdx(newPos, newColSize)
				newCol.SetVec(newIdx, oldCol.AtVec(oldIdx))
			})
		})
	}
	m.size = newColSize
}

func (m *MatColPicker) Pick(data *mat.Dense) *mat.Dense {
	_, c := data.Dims()
	searchStride := append([]int{}, m.size...)
	searchStride[m.pickDim] = 1
	retData := mat.NewDense(common.IntsProd(searchStride)*c, m.size[m.pickDim], nil)
	m.PickTo(retData, data)
	return retData
}
