package cu

import (
	"pneuma/common"
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
	fSize      int
	e          *Engine
	scalars    []float64
	scalarsPtr cu.DevicePtr
	datas      []mat.Matrix
	dataPtrs   []cu.DevicePtr
}

func NewMatCaltor(e *Engine) *MatCaltor {
	fSize := int(unsafe.Sizeof(0.0))
	scalars := make([]float64, 2)
	scalarsPtr, err := e.Alloc(int64(fSize) * 2)
	if err != nil {
		panic(err)
	}
	return &MatCaltor{e: e, fSize: fSize, scalars: scalars, scalarsPtr: scalarsPtr}
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

func (d *MatCaltor) DeviceDataByPtr(ptr cu.DevicePtr, size int) []float64 {
	var data []float64
	sh := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sh.Data = ptr.Uintptr()
	sh.Len = size
	sh.Cap = size
	return data
}

func (d *MatCaltor) DeviceDataByIdx(idx int) []float64 {
	r, c := d.datas[idx].Dims()
	size := r * c
	ptr := d.dataPtrs[idx]
	return d.DeviceDataByPtr(ptr, size)
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

func (d *MatCaltor) CopyTo(datas ...mat.Matrix) {
	for i := 0; i < len(datas); i++ {
		idx := d.Idx(datas[i])
		data := d.HostData(datas[i])
		if idx >= 0 {
			err := d.e.CopyToDevice(d.dataPtrs[idx], d.Pointer(data), int64(len(data)*d.fSize))
			if err != nil {
				panic(err)
			}
			continue
		}
		p, err := d.e.AllocAndCopy(d.Pointer(data), int64(len(data)*d.fSize))
		if err != nil {
			panic(err)
		}
		d.dataPtrs = append(d.dataPtrs, p)
		d.datas = append(d.datas, datas[i])
	}
}

func (d *MatCaltor) CopyBack(datas ...mat.Matrix) {
	for i := 0; i < len(datas); i++ {
		idx := d.Idx(datas[i])
		data := d.HostData(d.datas[idx])
		dst := d.Pointer(data)
		src := d.dataPtrs[idx]
		err := d.e.CopyToHost(dst, src, int64(len(data)*d.fSize))
		if err != nil {
			panic(err)
		}
	}
}

func (d *MatCaltor) CopyInDevice(dst, src mat.Matrix) {
	dstIdx := d.Idx(dst)
	srcIdx := d.Idx(src)
	hostData := d.HostData(dst)
	dstDev := d.dataPtrs[dstIdx]
	srcDev := d.dataPtrs[srcIdx]
	err := d.e.CopyInDevice(dstDev, srcDev, int64(len(hostData)*d.fSize))
	if err != nil {
		panic(err)
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
	d.MulKSpan(dst, am, bm, za, zb, 256)
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
		fun := func() error {
			d.e.blas.Dgemm(tA, tB, m, n, rk, 1, rpaf, lda, rpbf, ldb, 1, pcf, ldc)
			return d.e.blas.Err()
		}
		err := d.e.blas.Do(fun)
		if err != nil {
			panic(err)
		}
	}
}

func (d *MatCaltor) Add(dst, src mat.Matrix) {
	d.AddScaled(dst, 1, src)
}

func (d *MatCaltor) AddScaledSliceInc(dst mat.Matrix, alpha float64, src mat.Matrix, dstFrom, dstInc, srcFrom, srcInc, length int) {
	fun := func() error {
		y := d.DeviceData(dst)[dstFrom:]
		x := d.DeviceData(src)[srcFrom:]
		xinc := srcInc
		yinc := dstInc
		d.e.blas.Daxpy(length, alpha, x, xinc, y, yinc)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) AccRow(stride, length int, cb func(dstFrom, dstInc, srcFrom, srcInc, length int)) {
	for i := 0; i < length; i += stride {
		cb(0, 1, i, 1, stride)
	}
}

func (d *MatCaltor) AccCol(srcR, srcC int, cb func(dstFrom, dstInc, srcFrom, srcInc, length int)) {
	r, c := srcR, srcC
	for j := 0; j < c; j++ {
		cb(0, 1, j, c, r)
	}
}

func (d *MatCaltor) AddScaledOneByRow(dst mat.Matrix, alpha float64, rows mat.Matrix) {
	r, c := dst.Dims()
	dstLen := r * c
	r, c = rows.Dims()
	srcLen := r * c
	d.AccRow(dstLen, srcLen, func(dstFrom, dstInc, srcFrom, srcInc, length int) {
		d.AddScaledSliceInc(dst, alpha, rows, dstFrom, dstInc, srcFrom, srcInc, length)
	})
}

func (d *MatCaltor) AddScaledRowByOne(dst mat.Matrix, alpha float64, one mat.Matrix) {
	r, c := dst.Dims()
	dstLen := r * c
	r, c = one.Dims()
	srcLen := r * c
	d.AccRow(srcLen, dstLen, func(srcFrom, srcInc, dstFrom, dstInc, length int) {
		d.AddScaledSliceInc(dst, alpha, one, dstFrom, dstInc, srcFrom, srcInc, length)
	})
}

func (d *MatCaltor) AddScaledOneByCol(dst mat.Matrix, alpha float64, cols mat.Matrix) {
	r, c := cols.Dims()
	d.AccCol(r, c, func(dstFrom, dstInc, srcFrom, srcInc, length int) {
		d.AddScaledSliceInc(dst, alpha, cols, dstFrom, dstInc, srcFrom, srcInc, length)
	})
}

func (d *MatCaltor) AddScaledColByOne(dst mat.Matrix, alpha float64, one mat.Matrix) {
	r, c := dst.Dims()
	d.AccCol(r, c, func(srcFrom, srcInc, dstFrom, dstInc, length int) {
		d.AddScaledSliceInc(dst, alpha, one, dstFrom, dstInc, srcFrom, srcInc, length)
	})
}

func (d *MatCaltor) AddSliceInc(dst, src mat.Matrix, dstFrom, dstInc, srcFrom, srcInc, length int) {
	d.AddScaledSliceInc(dst, 1, src, dstFrom, dstInc, srcFrom, srcInc, length)
}

func (d *MatCaltor) AddSlice(dst, src mat.Matrix, dstFrom, srcFrom, length int) {
	d.AddSliceInc(dst, src, dstFrom, 1, srcFrom, 1, length)
}

func (d *MatCaltor) AddScaled(dst mat.Matrix, alpha float64, src mat.Matrix) {
	fun := func() error {
		pyf := d.DeviceData(dst)
		pxf := d.DeviceData(src)
		d.e.blas.Daxpy(len(pyf), alpha, pxf, 1, pyf, 1)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) ScaleSlice(alpha float64, data mat.Matrix, from, length int) {
	fun := func() error {
		pf := d.DeviceData(data)[from:]
		d.e.blas.Dscal(length, alpha, pf, 1)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) Scale(alpha float64, data mat.Matrix) {
	d.ScaleSlice(alpha, data, 0, len(d.HostData(data)))
}

func (d *MatCaltor) NormSliceInc(dst mat.Matrix, src mat.Matrix, dstFrom, srcFrom, srcInc, length int) {
	fun := func() error {
		y := d.DeviceData(dst)
		x := d.DeviceData(src)[srcFrom:]
		xinc := srcInc
		y[dstFrom] = d.e.blas.Dnrm2(length, x, xinc)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) NormOneByRow(dst, rows mat.Matrix) {
	r, c := rows.Dims()
	srcLen := r * c
	idx := 0
	d.AccRow(c, srcLen, func(_, _, srcFrom, srcInc, length int) {
		d.NormSliceInc(dst, rows, idx, srcFrom, srcInc, length)
		idx++
	})
}

func (d *MatCaltor) DotSliceInc(a, b mat.Matrix, dstFrom, aFrom, aInc, bFrom, bInc, length int) float64 {
	ret := 0.0
	fun := func() error {
		aDev := d.DeviceData(a)[aFrom:]
		bDev := d.DeviceData(b)[bFrom:]
		ret = d.e.blas.Ddot(length, aDev, aInc, bDev, bInc)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
	return ret
}

func (d *MatCaltor) DotRowByRowToHost(dstInHost *mat.VecDense, rowsA, rowsB mat.Matrix) {
	r, c := rowsA.Dims()
	srcLen := r * c
	idx := 0
	d.AccRow(c, srcLen, func(_, _, srcFrom, srcInc, length int) {
		val := d.DotSliceInc(rowsA, rowsB, idx, srcFrom, srcInc, srcFrom, srcInc, length)
		dstInHost.SetVec(idx, val)
		idx++
	})
}

func (d *MatCaltor) MulElemColByOneHost(dst mat.Matrix, oneInHost *mat.VecDense) {
	r, c := dst.Dims()
	dstLen := r * c
	idx := 0
	d.AccRow(c, dstLen, func(srcFrom, srcInc, dstFrom, dstInc, length int) {
		d.ScaleSlice(oneInHost.AtVec(idx), dst, dstFrom, length)
		idx++
	})
}

func (d *MatCaltor) OuterSliceInc(dst, a, b mat.Matrix, aFrom, aInc, bFrom, bInc, length int) {
	m, n := dst.Dims()
	fun := func() error {
		aDev := d.DeviceData(a)[aFrom:]
		bDev := d.DeviceData(b)[bFrom:]
		yDev := d.DeviceData(dst)
		d.e.blas.Dger(m, n, 1, aDev, aInc, bDev, bInc, yDev, length)
		return d.e.blas.Err()
	}
	err := d.e.blas.Do(fun)
	if err != nil {
		panic(err)
	}
}

func (d *MatCaltor) SVD(a, u, s, v mat.Matrix) {
	m, n := a.Dims()
	var lw int
	fun := func() error {
		lw = d.e.dn.DgesvdWork(m, n)
		return d.e.dn.Err()
	}
	err := d.e.ctx.Do(fun)
	if err != nil {
		panic(err)
	}
	w := mat.NewVecDense(lw, nil)
	rw := mat.NewVecDense(common.IntsMin(m, n)-1, nil)
	iDev, err := d.e.Alloc(int64(unsafe.Sizeof(0)))
	if err != nil {
		panic(err)
	}
	d.CopyTo(w, rw)
	la := m
	lu := m
	lv := n
	fun = func() error {
		aDev := d.DeviceData(a)
		sDev := d.DeviceData(s)
		uDev := d.DeviceData(u)
		vDev := d.DeviceData(v)
		wDev := d.DeviceData(w)
		rwDev := d.DeviceData(rw)
		var devInfo []int32
		devInfoSlice := (*reflect.SliceHeader)(unsafe.Pointer(&devInfo))
		devInfoSlice.Data = iDev.Uintptr()
		devInfoSlice.Len = 1
		devInfoSlice.Cap = 1
		d.e.dn.Dgesvd(m, n, aDev, la, sDev, uDev, lu, vDev, lv, wDev, lw, rwDev, devInfo)
		return d.e.dn.Err()
	}
	err = d.e.ctx.Do(fun)
	if err != nil {
		panic(err)
	}
	err = d.e.Free(iDev)
	if err != nil {
		panic(err)
	}
	d.Clear(w, rw)
}

func (d *MatCaltor) PCA(dst *mat.Dense, src mat.Matrix) {
	r, c := src.Dims()
	k, _ := dst.Dims()
	e := mat.NewVecDense(r, nil)
	alpha := 1.0 / float64(c)
	xsube := mat.DenseCopyOf(src)
	d.CopyTo(xsube, e)
	d.AddScaledOneByCol(e, alpha, src)
	d.AddScaledColByOne(xsube, -1, e)

	covLen := common.IntsMin(r, c)
	covMat := mat.NewDense(covLen, covLen, nil)
	svdS := mat.NewVecDense(covLen, nil)
	svdU := mat.NewDense(covLen, covLen, nil)
	svdV := mat.NewDense(covLen, covLen, nil)
	d.CopyTo(covMat, svdS, svdU, svdV)
	d.Mul(covMat, xsube, xsube, false, true)
	d.Scale(alpha, covMat)
	d.SVD(covMat, svdU, svdS, svdV)
	d.CopyBack(svdS, svdU)
	d.Clear(xsube, e, covMat, svdS, svdU, svdV)
	eigVecs := make([][][]float64, covLen)
	for i := 0; i < covLen; i++ {
		eigVecs[i] = [][]float64{
			svdU.RawRowView(i),
			{svdS.AtVec(i)},
		}
	}
	eigMat := mat.NewDense(k, r, nil)
	for i := 0; i < k; i++ {
		eigMat.SetRow(i, eigVecs[i][0])
	}
	d.CopyTo(eigMat)
	d.Mul(dst, eigMat, src, false, false)
	d.Clear(eigMat)
}
