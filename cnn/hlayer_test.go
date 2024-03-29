package cnn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestHLayerConv(t *testing.T) {
	layer := NewHLayerConv(
		ConvKernalParam{
			[]int{2, 2, 3},
			[]int{2, 2},
			ConvKernalPadFit,
		},
	)
	layer.InitSize([]int{4, 3, 2})
	data := mat.NewVecDense(24, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
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
		ConvKernalParam{
			[]int{3, 3, 1},
			[]int{1, 1},
			ConvKernalPadAll,
		},
	)
	layer.InitSize([]int{2, 2, 1})
	data := mat.NewVecDense(4, []float64{
		1, 2,
		3, 4,
	})
	tar := mat.NewDense(4, 1, []float64{
		5, 5,
		5, 5,
	})
	layer.W.Apply(func(i, j int, v float64) float64 {
		return 0.5
	}, layer.W)
	x := mat.NewDense(data.Len(), 1, data.RawVector().Data)
	y := layer.Forward(x)
	if !mat.Equal(y, tar) {
		t.Fatalf("forward not right need:\n%v but:\n%v\n", mat.Formatted(tar), mat.Formatted(y))
	}
	dy := mat.NewDense(4, 1, []float64{
		1, 1, 1, 1,
	})
	dx := layer.Backward(dy)
	dxr, dxc := dx.Dims()
	if dxr != data.Len() || dxc != 1 {
		t.Fatalf("backward not right need:r:%d,c:%d but:r:%d,c:%d\ny:\n%v\n", data.Len(), 1, dxr, dxc, mat.Formatted(dx))
	}
}
func TestHLayerMaxPooling(t *testing.T) {
	layer := NewHLayerMaxPooling(
		ConvKernalParam{
			[]int{2, 2},
			[]int{2, 2},
			ConvKernalPadFit,
		},
	)
	layer.InitSize([]int{4, 3, 2})
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
	dy := mat.NewDense(8, 1, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	dx := layer.Backward(dy)
	dxtar := mat.NewDense(24, 1, []float64{
		0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 4,
		0, 0, 0, 0, 0, 0,
		0, 0, 5, 6, 7, 8,
	})
	if !mat.Equal(dx, dxtar) {
		t.Fatalf("maxpooling backword wrong need:\n%v\nbut:\n%v\n", dxtar, dx)
	}
}
