package cu

import (
	"gonum.org/v1/gonum/mat"
)

type ConvCalter struct {
	*DenseCaltorCU
	packX *mat.Dense
	w     *mat.Dense
}

func NewConvCalter(eng *Engine) *ConvCalter {
	ret := &ConvCalter{}
	ret.DenseCaltorCU = NewDenseCaltorCU(eng)
	return ret
}

func (c *ConvCalter) Forward(packX, packY, w, b *mat.Dense) {
	c.Start(packX, packY, w, b)
	c.Mul(packY, packX, w, false, false)
	br, bc := b.Dims()
	blen := br * bc
	for j := 0; j < len(packY.RawMatrix().Data); j += blen {
		c.AddSlice(packY, b, j, 0, blen)
	}
	c.End(packY)
	c.Clear(packX, w)
	c.packX = packX
	c.w = w
}

func (c *ConvCalter) Backward(packDx, packDy, dw, db *mat.Dense) {
	c.Start(packDx, packDy, dw, db)
	c.Mul(dw, c.packX, packDy, true, false)
	br, bc := db.Dims()
	blen := br * bc
	for j := 0; j < len(packDy.RawMatrix().Data); j += blen {
		c.AddSlice(db, packDy, 0, j, blen)
	}
	c.Mul(packDx, packDy, c.w, false, true)
	c.End(packDx, dw, db)
	c.Clear()
	c.packX = nil
	c.w = nil
}
