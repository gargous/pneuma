package cu

import (
	"unsafe"

	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
)

type Engine struct {
	ctx *cu.Ctx
	*cublas.Standard
}

func NewEngine() *Engine {
	ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	bla := cublas.New(cublas.WithContext(ctx))
	ret := &Engine{
		ctx:      ctx,
		Standard: bla,
	}
	return ret
}

func (e *Engine) Alloc(size int64) (cu.DevicePtr, error) {
	return e.ctx.MemAllocManaged(size, cu.AttachGlobal)
}

func (e *Engine) AllocAndCopy(p unsafe.Pointer, size int64) (cu.DevicePtr, error) {
	dst, err := e.ctx.MemAllocManaged(size, cu.AttachGlobal)
	if err != nil {
		return dst, err
	}
	e.ctx.MemcpyHtoD(dst, p, size)
	return dst, e.ctx.Error()
}

func (e *Engine) CopyTo(dst cu.DevicePtr, src unsafe.Pointer, size int64) error {
	e.ctx.MemcpyHtoD(dst, src, size)
	return e.ctx.Error()
}

func (e *Engine) CopyBack(dst unsafe.Pointer, src cu.DevicePtr, size int64) error {
	e.ctx.MemcpyDtoH(dst, src, size)
	return e.ctx.Error()
}

func (e *Engine) Free(p cu.DevicePtr) error {
	e.ctx.MemFree(p)
	return e.ctx.Error()
}

/*
func NewEngine() *Engine {
	//ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	bla := cublas.New()
	ret := &Engine{
		Standard: bla,
		//ctx:      ctx,
	}
	return ret
}

func (e *Engine) Alloc(size int64) (cu.DevicePtr, error) {
	return cu.MemAlloc(size)
}

func (e *Engine) AllocAndCopy(p unsafe.Pointer, size int64) (cu.DevicePtr, error) {
	return cu.AllocAndCopy(p, size)
}

func (e *Engine) CopyBack(dst unsafe.Pointer, src cu.DevicePtr, size int64) error {
	return cu.MemcpyDtoH(dst, src, size)
}

func (e *Engine) Free(p cu.DevicePtr) error {
	return cu.MemFree(p)
}
*/
