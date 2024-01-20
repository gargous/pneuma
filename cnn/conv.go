package cnn

import (
	"fmt"
	"pneuma/common"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

func paddingCnt(size, core, stride int, padding ConvKernalPadding) (lp, rp, slip int) {
	var nopadSize int
	if padding == ConvKernalPadAll {
		slip = (size-1)/stride + 1
		rp = core / 2
		lp = rp
		return
	} else {
		if core > size {
			panic(fmt.Sprintf("paddingCnt need core bigger than size, now core=%d, size=%d", core, size))
		}
		slip = (size-core)/stride + 1
		nopadSize := (slip-1)*stride + core
		if nopadSize == size {
			return
		}
	}
	p := 0
	switch padding {
	case ConvKernalPadFit:
		p = nopadSize + stride - size
		slip += 1
	case ConvKernalPadNo:
		p = nopadSize - size
	}
	lp = p / 2
	rp = p - lp
	return
}

type ConvKernalPadding int16

const (
	ConvKernalPadNo ConvKernalPadding = iota
	ConvKernalPadFit
	ConvKernalPadAll
)

type ConvKernalParam struct {
	size    []int
	stride  []int
	padding ConvKernalPadding
}

func NewConvKParam(size, stride []int, padding ConvKernalPadding) ConvKernalParam {
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
		if i == dim-1 {
			ret.paddingLeft[i] = 0
			slips[i] = 1
			ret.fitSize[i] = inputSize[i]
		} else {
			pl, pr, slip := paddingCnt(iinp, icor, istr, param.padding)
			ret.paddingLeft[i] = pl
			slips[i] = slip
			ret.fitSize[i] = inputSize[i] + pl + pr
		}
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

type MatPicker struct {
	size    []int
	pickDim int
	dstDim  int
	srcDim  int
}

func NewMatPicker(size []int, pickDim int) *MatPicker {
	return &MatPicker{
		size:    size,
		pickDim: pickDim,
	}
}

func (m *MatPicker) SetMotion(dstDim, srcDim int) {
	m.dstDim = dstDim
	m.srcDim = srcDim
}

func (m *MatPicker) getter(data *mat.Dense) (gridData []float64, getter func(dst []float64, i int, a mat.Matrix) []float64) {
	dr, dc := data.Dims()
	switch m.srcDim {
	case 0:
		getter = mat.Row
		gridData = make([]float64, dc)
	case 1:
		getter = mat.Col
		gridData = make([]float64, dr)
	}
	return
}

func (m *MatPicker) setter(data *mat.Dense) func(j int, src []float64) {
	switch m.dstDim {
	case 0:
		return data.SetRow
	case 1:
		return data.SetCol
	}
	return nil
}

func (m *MatPicker) gridCnt(data *mat.Dense) (cnt int) {
	dr, dc := data.Dims()
	switch m.srcDim {
	case 0:
		cnt = dr
	case 1:
		cnt = dc
	}
	return
}

func (m *MatPicker) PickTo(dst, data *mat.Dense) {
	gridCnt := m.gridCnt(data)
	gridData, getter := m.getter(data)
	setter := m.setter(dst)
	gridC := common.IntsProd(m.size[m.pickDim:])
	gridR := common.IntsProd(m.size[:m.pickDim])
	gridCutC := common.IntsProd(m.size[m.pickDim+1:])
	gridCutR := gridR
	gridCutCnt := m.size[m.pickDim]
	gridCutStacks := make([]*mat.Dense, gridCutCnt)
	gridCSR := gridCutR
	gridCSC := gridCutC * gridCnt
	for j := 0; j < gridCnt; j++ {
		grid := mat.NewDense(gridR, gridC, getter(gridData, j, data))
		gcsIdx := j * gridCutC
		for gcj := 0; gcj < gridCutCnt; gcj++ {
			gcIdx := gcj * gridCutC
			gridCut := grid.Slice(0, gridCutR, gcIdx, gcIdx+gridCutC).(*mat.Dense)
			gcs := gridCutStacks[gcj]
			if gcs == nil {
				gcs = mat.NewDense(gridCSR, gridCSC, nil)
			}
			gcsSlice := gcs.Slice(0, gridCSR, gcsIdx, gcsIdx+gridCutC).(*mat.Dense)
			gcsSlice.Copy(gridCut)
			gridCutStacks[gcj] = gcs
		}
	}
	for j := 0; j < gridCutCnt; j++ {
		setter(j, gridCutStacks[j].RawMatrix().Data)
	}
	m.size[m.pickDim] = gridCnt
}

func (m *MatPicker) Pick(data *mat.Dense) (retData *mat.Dense) {
	searchStride := append([]int{}, m.size...)
	searchStride[m.pickDim] = 1
	cnt := m.gridCnt(data)
	switch m.dstDim {
	case 0:
		retData = mat.NewDense(m.size[m.pickDim], common.IntsProd(searchStride)*cnt, nil)
	case 1:
		retData = mat.NewDense(common.IntsProd(searchStride)*cnt, m.size[m.pickDim], nil)
	}
	m.PickTo(retData, data)
	return retData
}
