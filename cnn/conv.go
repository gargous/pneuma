package cnn

import (
	"fmt"

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
	_recuRange(size, stride, nil, cb)
}

func _recuRange(size, stride []int, pos []int, cb func(pos []int)) {
	if len(size) == 0 {
		cb(pos)
		return
	}
	for i := 0; i < size[0]; i += stride[0] {
		_recuRange(size[1:], stride[1:], append(pos, i), cb)
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
	for i := 0; i < len(a); i++ {
		ret[i] = a[i] + b[i]
	}
	return ret
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

func (c *ConvPacker) Pack(data *mat.VecDense) *mat.Dense {
	ret := mat.NewDense(c.slipCntSum, c.coreSizeSum, nil)
	cnt := c.orgSize[len(c.orgSize)-1]
	step := len(c.orgSize) - 2
	fitData := data
	if !intsEqual(c.orgSize, c.fitSize) {
		fitData = mat.NewVecDense(c.fitSizeSum, nil)
		padStep, pad := c.paddingLeft[:step], c.paddingLeft[step]
		offset := intsMin([]int{c.orgSize[step], c.fitSize[step]}) * cnt
		orgStepSize := c.orgSize[:step]
		fitStepSize := c.fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			fitStepPos := intsAdd(orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgPad, fitPad := 0, 0
			if pad < 0 {
				orgPad -= pad
			} else {
				fitPad += pad
			}
			orgIdx := posIdx(append(orgStepPos, orgPad, 0), c.orgSize)
			fitIdx := posIdx(append(fitStepPos, fitPad, 0), c.fitSize)
			fitSlice := fitData.SliceVec(fitIdx, fitIdx+offset).(*mat.VecDense)
			orgSlice := data.SliceVec(orgIdx, orgIdx+offset)
			fitSlice.CopyVec(orgSlice)
		})
	}
	slipRowIdx := 0
	recuRange(intsAddConst(1, intsSub(c.fitSize, c.coreSize)), c.stride, func(startPos []int) {
		slipRow := ret.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		coreStepSize := c.coreSize[:step]
		coreStepOffset := c.coreSize[step] * cnt
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			fitStepPos := intsAdd(coreStepPos, startStepPos)
			fitIdx := posIdx(append(fitStepPos, startStepOffset, 0), c.fitSize)
			slipIdx := posIdx(append(coreStepPos, 0, 0), c.coreSize)
			fitRow := fitData.SliceVec(fitIdx, fitIdx+coreStepOffset)
			slipRow.SliceVec(slipIdx, slipIdx+coreStepOffset).(*mat.VecDense).CopyVec(fitRow)
		})
	})
	return ret
}

func (c *ConvPacker) UnPack(data *mat.Dense) *mat.VecDense {
	fitData := mat.NewVecDense(c.fitSizeSum, nil)
	cnt := c.orgSize[len(c.orgSize)-1]
	step := len(c.orgSize) - 2
	slipRowIdx := 0
	recuRange(intsAddConst(1, intsSub(c.fitSize, c.coreSize)), c.stride, func(startPos []int) {
		slipRow := data.RowView(slipRowIdx).(*mat.VecDense)
		slipRowIdx += 1
		startStepPos := startPos[:step]
		startStepOffset := startPos[step]
		coreStepSize := c.coreSize[:step]
		coreStepOffset := c.coreSize[step] * cnt
		recuRange(coreStepSize, nil, func(coreStepPos []int) {
			fitStepPos := intsAdd(coreStepPos, startStepPos)
			fitIdx := posIdx(append(fitStepPos, startStepOffset, 0), c.fitSize)
			slipIdx := posIdx(append(coreStepPos, 0, 0), c.coreSize)
			fitRow := fitData.SliceVec(fitIdx, fitIdx+coreStepOffset).(*mat.VecDense)
			fitRow.AddVec(fitRow, slipRow.SliceVec(slipIdx, slipIdx+coreStepOffset))
		})
	})
	retData := fitData
	if !intsEqual(c.orgSize, c.fitSize) {
		retData = mat.NewVecDense(c.orgSizeSum, nil)
		padStep, pad := c.paddingLeft[:step], c.paddingLeft[step]
		offset := intsMin([]int{c.orgSize[step], c.fitSize[step]}) * cnt
		orgStepSize := c.orgSize[:step]
		fitStepSize := c.fitSize[:step]
		recuRange(orgStepSize, nil, func(orgStepPos []int) {
			fitStepPos := intsAdd(orgStepPos, padStep)
			if !sizeBound(fitStepPos, fitStepSize) {
				return
			}
			orgPad, fitPad := 0, 0
			if pad < 0 {
				orgPad -= pad
			} else {
				fitPad += pad
			}
			orgIdx := posIdx(append(orgStepPos, orgPad, 0), c.orgSize)
			fitIdx := posIdx(append(fitStepPos, fitPad, 0), c.fitSize)
			retSlice := retData.SliceVec(orgIdx, orgIdx+offset).(*mat.VecDense)
			fitSlice := fitData.SliceVec(fitIdx, fitIdx+offset)
			retSlice.CopyVec(fitSlice)
		})
	}
	return retData
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
	for i := 0; i < c; i++ {
		oldCol := data.ColView(i)
		recuRange(m.size, searchStride, func(startPos []int) {
			newCol := dst.ColView(startPos[m.pickDim]).(*mat.VecDense)
			recuRange(m.size, rangStride, func(rangPos []int) {
				pos := intsAdd(startPos, rangPos)
				oldIdx := posIdx(pos, m.size)
				newPos := append(append(pos[:m.pickDim], i), pos[m.pickDim+1:]...)
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
