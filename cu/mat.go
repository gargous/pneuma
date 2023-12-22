package cu

import (
	"reflect"
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/cu"
)

type IMatCaltor interface {
	Idx(data mat.Matrix) int
	Pointer(a []float64) unsafe.Pointer
	DeviceDataByIdx(idx int) []float64
	HostData(data mat.Matrix) []float64
	Start(datas ...mat.Matrix)
	End(datas ...mat.Matrix)
	Clear(datas ...mat.Matrix)

	AddScaled(dst mat.Matrix, alpha float64, src mat.Matrix)
}

type MatCaltor struct {
	e        *Engine
	datas    []mat.Matrix
	dataPtrs []cu.DevicePtr
}

func NewMatCaltor(e *Engine) *MatCaltor {
	return &MatCaltor{e: e}
}

func (d *MatCaltor) IdxOf(data mat.Matrix, datas []mat.Matrix) int {
	for i := 0; i < len(datas); i++ {
		if datas[i] == data {
			return i
		}
	}
	return -1
}

func (d *MatCaltor) Idx(data mat.Matrix) int {
	return d.IdxOf(data, d.datas)
}

func (d *MatCaltor) Pointer(a []float64) unsafe.Pointer {
	return unsafe.Pointer(&a[0])
}

func (d *MatCaltor) DeviceDataByIdx(idx int) []float64 {
	r, c := d.datas[idx].Dims()
	size := r * c
	ptr := d.dataPtrs[idx].Uintptr()
	var data []float64
	sh := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sh.Data = ptr
	sh.Len = size
	sh.Cap = size
	return data
}

func (d *MatCaltor) HostData(data mat.Matrix) []float64 {
	switch rdata := data.(type) {
	case *mat.Dense:
		return rdata.RawMatrix().Data
	case *mat.VecDense:
		return rdata.RawVector().Data
	}
	panic("invalid mat type")
}

func (d *MatCaltor) DeviceData(data mat.Matrix) []float64 {
	return d.DeviceDataByIdx(d.Idx(data))
}

func (d *MatCaltor) Start(datas ...mat.Matrix) {
	fsize := int(unsafe.Sizeof(0.0))
	for i := 0; i < len(datas); i++ {
		if d.Idx(datas[i]) >= 0 {
			continue
		}
		data := d.HostData(datas[i])
		p, err := d.e.AllocAndCopy(d.Pointer(data), int64(len(data)*fsize))
		if err != nil {
			panic(err)
		}
		d.dataPtrs = append(d.dataPtrs, p)
		d.datas = append(d.datas, datas[i])
	}
}

func (d *MatCaltor) End(datas ...mat.Matrix) {
	fsize := int(unsafe.Sizeof(0.0))
	for i := 0; i < len(datas); i++ {
		idx := d.Idx(datas[i])
		data := d.HostData(d.datas[idx])
		dst := d.Pointer(data)
		src := d.dataPtrs[idx]
		err := d.e.CopyBack(dst, src, int64(len(data)*fsize))
		if err != nil {
			panic(err)
		}
	}
}

func (d *MatCaltor) Clear(datas ...mat.Matrix) {
	for i := 0; i < len(datas); i++ {
		idx := d.Idx(datas[i])
		err := d.e.Free(d.dataPtrs[idx])
		if err != nil {
			panic(err)
		}
		d.dataPtrs = append(d.dataPtrs[:idx], d.dataPtrs[idx+1:]...)
		d.datas = append(d.datas[:idx], d.datas[idx+1:]...)
	}
}

func (d *MatCaltor) Mul(dst, am, bm mat.Matrix, za, zb bool) {
	d.MulKSpan(dst, am, bm, za, zb, 100)
}

func (d *MatCaltor) MulKSpan(dst, am, bm mat.Matrix, za, zb bool, kspan int) {
	a, b, c := d.Idx(am), d.Idx(bm), d.Idx(dst)
	m, k := d.datas[a].Dims()
	_, n := d.datas[b].Dims()
	tA, tB := blas.Trans, blas.Trans
	lda, ldb := 0, 0
	_, ldc := d.datas[c].Dims()
	switch {
	case !za && !zb:
		lda = k
		ldb = n
		tA, tB = blas.NoTrans, blas.NoTrans
		m, n = n, m
		lda, ldb = ldb, lda
		a, b = b, a
	case za && !zb:
		k, m = d.datas[a].Dims()
		lda = m
		ldb = n
		tA, tB = blas.Trans, blas.NoTrans
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		a, b = b, a
	case za && zb:
		k, m = d.datas[a].Dims()
		n, _ = d.datas[b].Dims()
		lda = m
		ldb = k
		tA, tB = blas.Trans, blas.Trans
		m, n = n, m
		lda, ldb = ldb, lda
		a, b = b, a
	case !za && zb:
		n, _ = d.datas[b].Dims()
		lda = k
		ldb = k
		tA, tB = blas.NoTrans, blas.Trans
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		a, b = b, a
	default:
		panic("Unreachable")
	}
	paf := d.DeviceDataByIdx(a)
	pbf := d.DeviceDataByIdx(b)
	pcf := d.DeviceDataByIdx(c)
	lk := 0
	for i := 0; i < k; i += kspan {
		rk := kspan
		rpaf := paf
		rpbf := pbf
		if i+kspan > k {
			rk = k - i
		}
		if tA == blas.NoTrans {
			rpaf = paf[lda*lk:]
		} else {
			rpaf = paf[lk:]
		}
		if tB == blas.NoTrans {
			rpbf = pbf[lk:]
		} else {
			rpbf = pbf[ldb*lk:]
		}
		lk += rk
		err := d.e.Do(func() error {
			d.e.Dgemm(tA, tB, m, n, rk, 1, rpaf, lda, rpbf, ldb, 1, pcf, ldc)
			return d.e.Err()
		})
		if err != nil {
			panic(err)
		}
	}

}

func (d *MatCaltor) Add(dst, src mat.Matrix) {
	d.AddScaled(dst, 1, src)
}

func (d *MatCaltor) AddSlice(dst, src mat.Matrix, dstFrom, srcFrom, length int) {
	err := d.e.Do(func() error {
		pyf := d.DeviceData(dst)[dstFrom:]
		pxf := d.DeviceData(src)[srcFrom:]
		d.e.Daxpy(length, 1, pxf, 1, pyf, 1)
		return d.e.Err()
	})
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) AddScaled(dst mat.Matrix, alpha float64, src mat.Matrix) {
	err := d.e.Do(func() error {
		pyf := d.DeviceData(dst)
		pxf := d.DeviceData(src)
		d.e.Daxpy(len(pyf), alpha, pxf, 1, pyf, 1)
		return d.e.Err()
	})
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) Scale(alpha float64, data mat.Matrix) {
	err := d.e.Do(func() error {
		pf := d.DeviceData(data)
		d.e.Dscal(len(d.HostData(data)), alpha, pf, 1)
		return d.e.Err()
	})
	if err != nil {
		panic(err)
	}
}
