package common

import "image"

func RecuRange(size, stride []int, cb func(pos []int)) {
	if stride == nil {
		stride = make([]int, len(size))
		IntsSetConst(1, stride)
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

func PosIdx(pos, size []int) int {
	idx := 0
	for i := 0; i < len(size); i++ {
		idx = IdxExpend(idx, pos[i], size[i])
	}
	return idx
}

func IdxExpend(oldIdx, newIdx, newSize int) int {
	return oldIdx*newSize + newIdx
}

func SizeBound(pos, size []int) bool {
	for i := 0; i < len(pos); i++ {
		if pos[i] < 0 || pos[i] >= size[i] {
			return false
		}
	}
	return true
}

func IntsEqual(a, b []int) bool {
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

func IntsMin(ints ...int) (ret int) {
	midx := -1
	for idx, v := range ints {
		if midx < 0 || ret > v {
			ret = v
			midx = idx
		}
	}
	return
}

func IntsProd(ints []int) (ret int) {
	ret = 1
	for _, v := range ints {
		ret *= v
	}
	return
}

func IntsSetConst(a int, b []int) {
	for i := 0; i < len(b); i++ {
		b[i] = a
	}
}

func IntsAdd(a, b []int) []int {
	ret := make([]int, len(a))
	IntsAddTo(ret, a, b)
	return ret
}

func IntsAddTo(src, a, b []int) {
	for i := 0; i < len(a); i++ {
		src[i] = a[i] + b[i]
	}
}

func IntsAddConst(a int, b []int) []int {
	ret := make([]int, len(b))
	IntsAddConstTo(ret, a, b)
	return ret
}

func IntsAddConstTo(src []int, a int, b []int) {
	for i := 0; i < len(b); i++ {
		src[i] = a + b[i]
	}
}

func IntsSub(a, b []int) []int {
	ret := make([]int, len(a))
	for i := 0; i < len(a); i++ {
		ret[i] = a[i] - b[i]
	}
	return ret
}

type Rectangle struct {
	image.Rectangle
	Area float64
}

func Rect(x0, y0, x1, y1 int) Rectangle {
	return rectByImR(image.Rect(x0, y0, x1, y1))
}

func rectByImR(r image.Rectangle) Rectangle {
	size := r.Size()
	return Rectangle{
		Rectangle: r,
		Area:      float64(size.X * size.Y),
	}
}

func (r Rectangle) IOU(o Rectangle) float64 {
	c := r.Union(o.Rectangle)
	rc := rectByImR(c)
	return rc.Area / (r.Area + o.Area - rc.Area)
}
