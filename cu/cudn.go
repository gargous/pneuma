package cu

// #include <cusolverDn.h>
import "C"
import (
	"errors"
	"sync"

	"gorgonia.org/cu"
)

func cudnStatus(x C.cusolverStatus_t) error {
	if x == C.CUSOLVER_STATUS_SUCCESS {
		return nil
	}
	if x == C.CUSOLVER_STATUS_INTERNAL_ERROR {
		return nil
	}
	return errors.New(cudnStatusString[x])
}

var cudnStatusString = map[C.cusolverStatus_t]string{
	C.CUSOLVER_STATUS_SUCCESS:                   "Success",
	C.CUSOLVER_STATUS_NOT_INITIALIZED:           "NotInitialized",
	C.CUSOLVER_STATUS_ALLOC_FAILED:              "AllocFailed",
	C.CUSOLVER_STATUS_INVALID_VALUE:             "InvalidValue",
	C.CUSOLVER_STATUS_ARCH_MISMATCH:             "ArchMismatch",
	C.CUSOLVER_STATUS_MAPPING_ERROR:             "MappingError",
	C.CUSOLVER_STATUS_EXECUTION_FAILED:          "ExecFailed",
	C.CUSOLVER_STATUS_INTERNAL_ERROR:            "InternalError",
	C.CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: "MatTypeNotSupported",
	C.CUSOLVER_STATUS_NOT_SUPPORTED:             "NoSupported",
	C.CUSOLVER_STATUS_ZERO_PIVOT:                "ZeroPivot",
	C.CUSOLVER_STATUS_INVALID_LICENSE:           "InvalidLicense",
}

type Cudn struct {
	h C.cusolverDnHandle_t
	e error
	cu.Context
	sync.Mutex
}

func NewCudn(ctx cu.Context) *Cudn {
	var handle C.cusolverDnHandle_t
	if err := cudnStatus(C.cusolverDnCreate(&handle)); err != nil {
		panic(err)
	}
	impl := &Cudn{
		h:       handle,
		Context: ctx,
	}
	return impl
}

func (impl *Cudn) Err() error { return impl.e }

func (impl *Cudn) Close() error {
	impl.Lock()
	defer impl.Unlock()

	var empty C.cusolverDnHandle_t
	if impl.h == empty {
		return nil
	}
	if err := cudnStatus(C.cusolverDnDestroy(impl.h)); err != nil {
		return err
	}
	impl.h = empty
	impl.Context = nil
	return nil
}

func (impl *Cudn) DgesvdWork(m, n int) int {
	var wl int32
	impl.e = cudnStatus(C.cusolverDnDgesvd_bufferSize(C.cusolverDnHandle_t(impl.h), C.int(m), C.int(n), (*C.int)(&wl)))
	return int(wl)
}

func (impl *Cudn) Dgesvd(m, n int, a []float64, la int, s []float64, u []float64, lu int, v []float64, lv int, w []float64, lw int, rw []float64, dev []int32) {
	jobU := 'A'
	jobV := 'A'
	impl.e = cudnStatus(C.cusolverDnDgesvd(C.cusolverDnHandle_t(impl.h), C.schar(jobU), C.schar(jobV),
		C.int(m), C.int(n), (*C.double)(&a[0]), C.int(la),
		(*C.double)(&s[0]),
		(*C.double)(&u[0]), C.int(lu),
		(*C.double)(&v[0]), C.int(lv),
		(*C.double)(&w[0]), C.int(lw), (*C.double)(&rw[0]),
		(*C.int)(&dev[0]),
	))
}
