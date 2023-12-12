package cnn

import "fmt"

func paddingCnt(size, core, stride int, padding bool) (lp, rp, slip int) {
	if core > size {
		panic(fmt.Sprintf("paddingCnt need core bigger than size, now core=%d, size=%d", core, size))
	}
	slip = (size - core) / stride
	nopadSize := slip*stride + core
	slip += 1
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
