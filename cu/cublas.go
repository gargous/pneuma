package cu

// #include <cublas_v2.h>
import "C"
import (
	"sync"

	"gonum.org/v1/gonum/blas"
	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
)

type CublasPointMode uint32

const (
	CublasPointModeHost   CublasPointMode = C.CUBLAS_POINTER_MODE_HOST
	CublasPointModeDevice CublasPointMode = C.CUBLAS_POINTER_MODE_DEVICE
)

func trans2cublasTrans(t blas.Transpose) C.cublasOperation_t {
	switch t {
	case blas.NoTrans:
		return cublas.NoTrans
	case blas.Trans:
		return cublas.Trans
	case blas.ConjTrans:
		return cublas.ConjTrans
	}
	panic("Unreachable")
}

var cublasStatusString = map[cublas.Status]string{
	cublas.Success:        "Success",
	cublas.NotInitialized: "NotInitialized",
	cublas.AllocFailed:    "AllocFailed",
	cublas.InvalidValue:   "InvalidValue",
	cublas.ArchMismatch:   "ArchMismatch",
	cublas.MappingError:   "MappingError",
	cublas.ExecFailed:     "ExecFailed",
	cublas.InternalError:  "InternalError",
	cublas.Unsupported:    "Unsupported",
	cublas.LicenceError:   "LicenceError",
}

func cublasStatus(x C.cublasStatus_t) error {
	err := cublas.Status(x)
	if err == cublas.Success {
		return nil
	}
	if err > cublas.LicenceError {
		return cublas.Unsupported
	}
	return err
}

func cublasStatusKnown(x C.cublasStatus_t) error {
	err := cublas.Status(x)
	if err == cublas.Success {
		return nil
	}
	if err == cublas.InternalError {
		return nil
	}
	if err > cublas.LicenceError {
		return cublas.Unsupported
	}
	return err
}

type Cublas struct {
	h C.cublasHandle_t
	e error

	cu.Context
	dataOnDev bool

	sync.Mutex
}

func NewCublas(ctx cu.Context) *Cublas {
	var handle C.cublasHandle_t
	if err := cublasStatus(C.cublasCreate(&handle)); err != nil {
		panic(err)
	}
	impl := &Cublas{
		h:       handle,
		Context: ctx,
	}
	if err := impl.SetPointerMode(CublasPointModeHost); err != nil {
		panic(err)
	}
	return impl
}

func (impl *Cublas) Err() error { return impl.e }

func (impl *Cublas) Close() error {
	impl.Lock()
	defer impl.Unlock()

	var empty C.cublasHandle_t
	if impl.h == empty {
		return nil
	}
	if err := cublasStatus(C.cublasDestroy(impl.h)); err != nil {
		return err
	}
	impl.h = empty
	impl.Context = nil
	return nil
}

func (impl *Cublas) SetPointerMode(mode CublasPointMode) error {
	return cublasStatus(C.cublasSetPointerMode(C.cublasHandle_t(impl.h), C.cublasPointerMode_t(mode)))
}

// y[i] += alpha * x[i] for all i
func (impl *Cublas) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	// declared at cublasgen.h:304:17 enum CUBLAS_STATUS { ... } cublasDaxpy ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = cublasStatus(C.cublasDaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY)))
}

// C = beta * C + alpha * A * B,
func (impl *Cublas) Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1376:17 enum CUBLAS_STATUS { ... } cublasDgemm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	impl.e = cublasStatus(C.cublasDgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

// Ddot computes the dot product of the two vectors
//
//	\sum_i x[i]*y[i]
func (impl *Cublas) Ddot(n int, x []float64, incX int, y []float64, incY int) (retVal float64) {
	// declared at cublasgen.h:194:17 enum CUBLAS_STATUS { ... } cublasDdot ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = cublasStatusKnown(C.cublasDdot(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&retVal)))
	return retVal
}

// sqrt(\sum_i x[i] * x[i]).
func (impl *Cublas) Dnrm2(n int, x []float64, incX int) float64 {
	// declared at cublasgen.h:143:17 enum CUBLAS_STATUS { ... } cublasDnrm2 ...
	if impl.e != nil {
		return 0
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return 0
	}
	var ret float64
	impl.e = cublasStatusKnown(C.cublasDnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&ret)))
	return ret
}

// x[i] *= alpha
func (impl *Cublas) Dscal(n int, alpha float64, x []float64, incX int) {
	// declared at cublasgen.h:251:17 enum CUBLAS_STATUS { ... } cublasDscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = cublasStatus(C.cublasDscal(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX)))
}

// y = alpha * A * x + beta * y if tA == blas.NoTrans
// y = alpha * A^T * x + beta * y if tA == blas.Trans or blas.ConjTrans
func (impl *Cublas) Dgbmv(tA blas.Transpose, m, n, kl, ku int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:634:17 enum CUBLAS_STATUS { ... } cublasDgbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	impl.e = cublasStatus(C.cublasDgbmv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}
