package cu

import (
	"reflect"
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/cu"
)

type DenseCaltorCU struct {
	e        *Engine
	datas    []*mat.Dense
	dataPtrs []cu.DevicePtr
}

func NewDenseCaltorCU(e *Engine) *DenseCaltorCU {
	return &DenseCaltorCU{e: e}
}

func (d *DenseCaltorCU) IdxOf(data *mat.Dense, datas []*mat.Dense) int {
	for i := 0; i < len(datas); i++ {
		if datas[i] == data {
			return i
		}
	}
	return -1
}

func (d *DenseCaltorCU) Idx(data *mat.Dense) int {
	return d.IdxOf(data, d.datas)
}

func (d *DenseCaltorCU) Pointer(a []float64) unsafe.Pointer {
	return unsafe.Pointer(&a[0])
}

func (d *DenseCaltorCU) DeviceDataByIdx(idx int) []float64 {
	size := len(d.datas[idx].RawMatrix().Data)
	ptr := d.dataPtrs[idx].Uintptr()
	var data []float64
	sh := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sh.Data = ptr
	sh.Len = size
	sh.Cap = size
	return data
}

func (d *DenseCaltorCU) DeviceData(data *mat.Dense) []float64 {
	return d.DeviceDataByIdx(d.Idx(data))
}

func (d *DenseCaltorCU) Start(datas ...*mat.Dense) {
	fsize := int(unsafe.Sizeof(0.0))
	for i := 0; i < len(datas); i++ {
		data := datas[i].RawMatrix().Data
		p, err := d.e.AllocAndCopy(d.Pointer(data), int64(len(data)*fsize))
		if err != nil {
			panic(err)
		}
		d.dataPtrs = append(d.dataPtrs, p)
		d.datas = append(d.datas, datas[i])
	}
}

func (d *DenseCaltorCU) End(datas ...*mat.Dense) {
	fsize := int(unsafe.Sizeof(0.0))
	for i := 0; i < len(datas); i++ {
		idx := d.Idx(datas[i])
		data := d.datas[idx].RawMatrix().Data
		dst := d.Pointer(data)
		src := d.dataPtrs[idx]
		err := d.e.CopyBack(dst, src, int64(len(data)*fsize))
		if err != nil {
			panic(err)
		}
	}
}

func (d *DenseCaltorCU) Clear(excluds ...*mat.Dense) {
	newPtrs := make([]cu.DevicePtr, 0)
	newDatas := make([]*mat.Dense, 0)
	for i := 0; i < len(d.dataPtrs); i++ {
		if d.IdxOf(d.datas[i], excluds) >= 0 {
			newPtrs = append(newPtrs, d.dataPtrs[i])
			newDatas = append(newDatas, d.datas[i])
		} else {
			err := d.e.Free(d.dataPtrs[i])
			if err != nil {
				panic(err)
			}
		}
	}
	d.dataPtrs = newPtrs
	d.datas = newDatas
}

func (d *DenseCaltorCU) Mul(dst, am, bm *mat.Dense, za, zb bool) {
	d.MulKSpan(dst, am, bm, za, zb, 100)
}

func (d *DenseCaltorCU) MulKSpan(dst, am, bm *mat.Dense, za, zb bool, kspan int) {
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

func (d *DenseCaltorCU) Add(dst, src *mat.Dense) {
	err := d.e.Do(func() error {
		pyf := d.DeviceData(dst)
		pxf := d.DeviceData(src)
		d.e.Daxpy(len(pyf), 1, pxf, 1, pyf, 1)
		return d.e.Err()
	})
	if err != nil {
		panic(err)
	}
}

func (d *DenseCaltorCU) AddSlice(dst, src *mat.Dense, dstFrom, srcFrom, length int) {
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

func (d *DenseCaltorCU) Scale(alpha float64, data *mat.Dense) {
	err := d.e.Do(func() error {
		pf := d.DeviceData(data)
		d.e.Dscal(len(data.RawMatrix().Data), alpha, pf, 1)
		return d.e.Err()
	})
	if err != nil {
		panic(err)
	}
}
