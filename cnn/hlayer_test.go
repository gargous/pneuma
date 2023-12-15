package cnn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestHLayerConv(t *testing.T) {
	layer := NewHLayerConv(
		[]int{4, 3, 2},
		[]int{2, 2, 3},
		[]int{2, 2},
		true,
	)
	data := mat.NewVecDense(24, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
	slip := layer.c.slipBuild(data)
	tarSlip := mat.NewDense(4, 8, []float64{
		1, 2, 3, 4, 7, 8, 9, 10,
		5, 6, 0, 0, 11, 12, 0, 0,
		13, 14, 15, 16, 19, 20, 21, 22,
		17, 18, 0, 0, 23, 24, 0, 0,
	})
	if !mat.Equal(slip, tarSlip) {
		t.Fatalf("slip build not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarSlip), mat.Formatted(slip))
	}
	slipRet := layer.c.slipRestore(slip)
	if !mat.Equal(slipRet, data) {
		t.Fatalf("slip restore not right need:\n%v\nbut:\n%v\n", data, slipRet)
	}
	x := mat.NewDense(data.Len(), 1, data.RawVector().Data)
	y := layer.Forward(x)
	yr, yc := y.Dims()
	if yr != 12 || yc != 1 {
		t.Fatalf("forward not right need:r:%d,c:%d but:r:%d,c:%d\ny:\n%v\n", 12, 1, yr, yc, mat.Formatted(y))
	}
	dy := mat.NewDense(12, 1, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	dx := layer.Backward(dy)
	dxr, dxc := dx.Dims()
	if dxr != data.Len() || dxc != 1 {
		t.Fatalf("backward not right need:r:%d,c:%d but:r:%d,c:%d\ny:\n%v\n", data.Len(), 1, dxr, dxc, mat.Formatted(dx))
	}
}
func TestHLayerConv2(t *testing.T) {
	layer := NewHLayerConv(
		[]int{4, 3, 2},
		[]int{2, 2, 3},
		[]int{1, 1},
		true,
	)
	data := mat.NewVecDense(24, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
	slip := layer.c.slipBuild(data)
	tarSlip := mat.NewDense(6, 8, []float64{
		1, 2, 3, 4, 7, 8, 9, 10,
		3, 4, 5, 6, 9, 10, 11, 12,
		7, 8, 9, 10, 13, 14, 15, 16,
		9, 10, 11, 12, 15, 16, 17, 18,
		13, 14, 15, 16, 19, 20, 21, 22,
		15, 16, 17, 18, 21, 22, 23, 24,
	})
	if !mat.Equal(slip, tarSlip) {
		t.Fatalf("slip build not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarSlip), mat.Formatted(slip))
	}
	slipRet := layer.c.slipRestore(slip)
	tarSlipRet := mat.NewVecDense(24, []float64{
		1, 2, 6, 8, 5, 6,
		14, 16, 36, 40, 22, 24,
		26, 28, 60, 64, 34, 36,
		19, 20, 42, 44, 23, 24,
	})
	if !mat.Equal(slipRet, tarSlipRet) {
		t.Fatalf("slip restore not right need:\n%v\nbut:\n%v\n", tarSlipRet, slipRet)
	}
	x := mat.NewDense(data.Len(), 1, data.RawVector().Data)
	y := layer.Forward(x)
	yr, yc := y.Dims()
	if yr != 18 || yc != 1 {
		t.Fatalf("forward not right need:r:%d,c:%d but:r:%d,c:%d\ny:\n%v\n", 18, 1, yr, yc, mat.Formatted(y))
	}
	dy := mat.NewDense(18, 1, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})
	dx := layer.Backward(dy)
	dxr, dxc := dx.Dims()
	if dxr != data.Len() || dxc != 1 {
		t.Fatalf("backward not right need:r:%d,c:%d but:r:%d,c:%d\ny:\n%v\n", data.Len(), 1, dxr, dxc, mat.Formatted(dx))
	}
}
func TestHLayerMaxPooling(t *testing.T) {
	layer := NewHLayerMaxPooling(
		[]int{4, 3, 2},
		[]int{2, 2},
		[]int{2, 2},
		true,
	)
	x := mat.NewDense(24, 1, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
	y := layer.Forward(x)
	ytar := mat.NewDense(8, 1, []float64{
		9, 10, 11, 12,
		21, 22, 23, 24,
	})
	if !mat.Equal(ytar, y) {
		t.Fatalf("maxpooling forward wrong need:\n%v\nbut:\n%v\n", ytar, y)
	}
	dy := mat.NewDense(8, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	dx := layer.Backward(dy)
	dxtar := mat.NewDense(24, 1, []float64{
		0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 4,
		0, 0, 0, 0, 0, 0,
		0, 0, 5, 6, 7, 8,
	})
	if !mat.Equal(ytar, y) {
		t.Fatalf("maxpooling backword wrong need:\n%v\nbut:\n%v\n", dxtar, dx)
	}
}
