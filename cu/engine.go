package cu

import (
	"unsafe"

	"gorgonia.org/cu"
)

type Engine struct {
	ctx  *cu.Ctx
	blas *Cublas
	dn   *Cudn
}

func NewEngine() *Engine {
	ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	bla := NewCublas(ctx)
	dn := NewCudn(ctx)
	ret := &Engine{
		ctx:  ctx,
		blas: bla,
		dn:   dn,
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

func (e *Engine) CopyInDevice(dst, src cu.DevicePtr, size int64) error {
	e.ctx.Memcpy(dst, src, size)
	return e.ctx.Error()
}

func (e *Engine) CopyToDevice(dst cu.DevicePtr, src unsafe.Pointer, size int64) error {
	e.ctx.MemcpyHtoD(dst, src, size)
	return e.ctx.Error()
}

func (e *Engine) CopyToHost(dst unsafe.Pointer, src cu.DevicePtr, size int64) error {
	e.ctx.MemcpyDtoH(dst, src, size)
	return e.ctx.Error()
}

func (e *Engine) Free(p cu.DevicePtr) error {
	e.ctx.MemFree(p)
	return e.ctx.Error()
}

func (e *Engine) Close() {
	e.blas.Close()
	e.dn.Close()
	e.ctx.Close()
}
