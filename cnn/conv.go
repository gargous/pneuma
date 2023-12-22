package cnn

import (
	"fmt"

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

func recuRange(size, stride []int, cb func(pos []int)) {
	if stride == nil {
		stride = make([]int, len(size))
		for i := 0; i < len(size); i++ {
			stride[i] = 1
		}
	}
	_recuRange(0, size, stride, make([]int, len(size)), cb)
}

func _recuRange(depth int, size, stride []int, pos []int, cb func(pos []int)) {
	if depth == len(size) {
		cb(pos)
		return
	}
	for i := 0; i < size[depth]; i += stride[depth] {
		pos[depth] = i
		_recuRange(depth+1, size, stride, pos, cb)
	}
}

func posIdx(pos, size []int) int {
	idx := 0
	for i := 0; i < len(size); i++ {
		idx = idxExpend(idx, pos[i], size[i])
	}
	return idx
}

func idxExpend(oldIdx, newIdx, newSize int) int {
	return oldIdx*newSize + newIdx
}

func sizeBound(pos, size []int) bool {
	for i := 0; i < len(pos); i++ {
		if pos[i] < 0 || pos[i] >= size[i] {
			return false
		}
	}
	return true
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func intsMin(ints []int) (ret int) {
	midx := -1
	for idx, v := range ints {
		if midx < 0 || ret > v {
			ret = v
			midx = idx
		}
	}
	return
}

func intsProd(ints []int) (ret int) {
	ret = 1
	for _, v := range ints {
		ret *= v
	}
	return
}

func intsAdd(a, b []int) []int {
	ret := make([]int, len(a))
	intsAddTo(ret, a, b)
	return ret
}

func intsAddTo(src, a, b []int) {
	for i := 0; i < len(a); i++ {
		src[i] = a[i] + b[i]
	}
}

func intsAddConst(a int, b []int) []int {
	ret := make([]int, len(b))
	for i := 0; i < len(b); i++ {
		ret[i] = a + b[i]
	}
	return ret
}

func intsSub(a, b []int) []int {
	ret := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		ret[i] = a[i] - b[i]
	}
	return ret
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
}

func NewConvPacker(inputSize, core, stride []int, padding bool) *ConvPacker {
	dim := len(inputSize)
	inpCnt := inputSize[len(inputSize)-1]
	ret := &ConvPacker{
		orgSize:     inputSize,
		coreSize:    append(core, inpCnt),
		stride:      append(stride, inpCnt),
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
		pl, pr, slip := paddingCnt(iinp, icor, istr, padding)
		ret.paddingLeft[i] = pl
		slips[i] = slip
		ret.fitSize[i] = inputSize[i] + pl + pr
	}
	ret.slipCnt = slips
	ret.fitSizeSum = intsProd(ret.fitSize)
	ret.orgSizeSum = intsProd(ret.orgSize)
	ret.slipCntSum = intsProd(slips)
	ret.coreSizeSum = intsProd(ret.coreSize)
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

func (c *ConvPacker) Pack(vec *mat.VecDense) *mat.Dense {
	ret := mat.NewDense(c.slipCntSum, c.coreSizeSum, nil)
	cnt := c.orgSize[len(c.orgSize)-1]
	step := len(c.orgSize) - 2
	fitVec := vec
	fitStepPos := make([]int, step)
	if !intsEqual(c.orgSize, c.fitSize) {
		fitVec = mat.NewVecDense(c.fitSizeSum, nil)
		padStep, pad := c.paddingLeft[:step], c.paddingLeft[step]
		offset := intsMin([]int{c.orgSize[step], c.fitSize[step]}) * cnt
		orgStepSize := c.orgSize[:step]
		fitStepSize := c.fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			intsAddTo(fitStepPos, orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgPad, fitPad := 0, 0
			if pad < 0 {
				orgPad -= pad
			} else {
				fitPad += pad
			}

			orgIdx := idxExpend(idxExpend(posIdx(orgStepPos, orgStepSize), orgPad, c.orgSize[step]), 0, c.orgSize[step+1])
			fitIdx := idxExpend(idxExpend(posIdx(fitStepPos, fitStepSize), fitPad, c.fitSize[step]), 0, c.fitSize[step+1])
			sliceVecCopy(fitVec, vec, fitIdx, orgIdx, offset)
		})
	}
	slipRowIdx := 0
	coreStepSize := c.coreSize[:step]
	coreStepOffset := c.coreSize[step] * cnt
	fitStepSize := c.fitSize[:step]
	recuRange(intsAddConst(1, intsSub(c.fitSize, c.coreSize)), c.stride, func(startPos []int) {
		slipRow := ret.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			intsAddTo(fitStepPos, coreStepPos, startStepPos)
			fitIdx := idxExpend(idxExpend(posIdx(fitStepPos, fitStepSize), startStepOffset, c.fitSize[step]), 0, c.fitSize[step+1])
			slipIdx := idxExpend(idxExpend(posIdx(coreStepPos, coreStepSize), 0, c.coreSize[step]), 0, c.coreSize[step+1])
			sliceVecCopy(slipRow, fitVec, slipIdx, fitIdx, coreStepOffset)
		})
	})
	return ret
}

func (c *ConvPacker) UnPack(vec *mat.Dense) *mat.VecDense {
	fitVec := mat.NewVecDense(c.fitSizeSum, nil)
	cnt := c.orgSize[len(c.orgSize)-1]
	step := len(c.orgSize) - 2
	slipRowIdx := 0
	coreStepSize := c.coreSize[:step]
	coreStepOffset := c.coreSize[step] * cnt
	fitStepSize := c.fitSize[:step]
	fitStepPos := make([]int, step)
	recuRange(intsAddConst(1, intsSub(c.fitSize, c.coreSize)), c.stride, func(startPos []int) {
		slipRow := vec.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			intsAddTo(fitStepPos, coreStepPos, startStepPos)
			fitIdx := idxExpend(idxExpend(posIdx(fitStepPos, fitStepSize), startStepOffset, c.fitSize[step]), 0, c.fitSize[step+1])
			slipIdx := idxExpend(idxExpend(posIdx(coreStepPos, coreStepSize), 0, c.coreSize[step]), 0, c.coreSize[step+1])
			fitRow := fitVec.SliceVec(fitIdx, fitIdx+coreStepOffset).(*mat.VecDense)
			fitRow.AddVec(fitRow, slipRow.SliceVec(slipIdx, slipIdx+coreStepOffset))
		})
	})
	retVec := fitVec
	if !intsEqual(c.orgSize, c.fitSize) {
		retVec = mat.NewVecDense(c.orgSizeSum, nil)
		padStep, pad := c.paddingLeft[:step], c.paddingLeft[step]
		offset := intsMin([]int{c.orgSize[step], c.fitSize[step]}) * cnt
		orgStepSize := c.orgSize[:step]
		fitStepSize := c.fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			intsAddTo(fitStepPos, orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgPad, fitPad := 0, 0
			if pad < 0 {
				orgPad -= pad
			} else {
				fitPad += pad
			}
			orgIdx := idxExpend(idxExpend(posIdx(orgStepPos, orgStepSize), orgPad, c.orgSize[step]), 0, c.orgSize[step+1])
			fitIdx := idxExpend(idxExpend(posIdx(fitStepPos, fitStepSize), fitPad, c.fitSize[step]), 0, c.fitSize[step+1])
			sliceVecCopy(retVec, fitVec, orgIdx, fitIdx, offset)
		})
	}
	return retVec
}

func (c *ConvPacker) FoldBatches(org, fld *mat.Dense, cb func(data *mat.VecDense) *mat.Dense) *mat.Dense {
	_, batch := org.Dims()
	fldRow, fldCol := fld.Dims()
	fldBatch := fldRow / batch
	for j := 0; j < batch; j++ {
		sliceFld := fld.Slice(j*fldBatch, j*fldBatch+fldBatch, 0, fldCol).(*mat.Dense)
		if cb == nil {
			sliceFld.Copy(org.ColView(j).(*mat.VecDense))
		} else {
			sliceFld.Copy(cb(org.ColView(j).(*mat.VecDense)))
		}
	}
	return fld
}

func (c *ConvPacker) UnfoldBatches(org, fld *mat.Dense, cb func(data *mat.Dense) *mat.VecDense) *mat.Dense {
	_, batch := org.Dims()
	fldRow, fldCol := fld.Dims()
	fldBatch := fldRow / batch
	for j := 0; j < batch; j++ {
		sliceFld := fld.Slice(j*fldBatch, j*fldBatch+fldBatch, 0, fldCol).(*mat.Dense)
		if cb == nil {
			org.SetCol(j, sliceFld.RawMatrix().Data)
		} else {
			org.SetCol(j, cb(sliceFld).RawVector().Data)
		}
	}
	return org
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
	rangStride := intsAddConst(1, make([]int, len(m.size)))
	rangStride[m.pickDim] = m.size[m.pickDim]
	newColSize := append([]int{}, m.size...)
	newColSize[m.pickDim] = c
	pos := make([]int, len(m.size))
	newPos := make([]int, len(m.size))
	for i := 0; i < c; i++ {
		oldCol := data.ColView(i)
		recuRange(m.size, searchStride, func(startPos []int) {
			newCol := dst.ColView(startPos[m.pickDim]).(*mat.VecDense)
			recuRange(m.size, rangStride, func(rangPos []int) {
				intsAddTo(pos, startPos, rangPos)
				oldIdx := posIdx(pos, m.size)
				copy(newPos, pos)
				newPos[m.pickDim] = i
				newIdx := posIdx(newPos, newColSize)
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
	retData := mat.NewDense(intsProd(searchStride)*c, m.size[m.pickDim], nil)
	m.PickTo(retData, data)
	return retData
}
