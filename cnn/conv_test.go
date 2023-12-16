package cnn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConvPacker1(t *testing.T) {
	packer := NewConvPacker(
		[]int{4, 3, 2},
		[]int{2, 2},
		[]int{1, 1},
		true,
	)
	data := mat.NewVecDense(24, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
	slip := packer.Pack(data)
	tarSlip := mat.NewDense(6, 8, []float64{
		1, 2, 3, 4, 7, 8, 9, 10,
		3, 4, 5, 6, 9, 10, 11, 12,
		7, 8, 9, 10, 13, 14, 15, 16,
		9, 10, 11, 12, 15, 16, 17, 18,
		13, 14, 15, 16, 19, 20, 21, 22,
		15, 16, 17, 18, 21, 22, 23, 24,
	})
	if !mat.Equal(slip, tarSlip) {
		t.Fatalf("slip pack not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarSlip), mat.Formatted(slip))
	}
	slipRet := packer.UnPack(slip)
	tarSlipRet := mat.NewVecDense(24, []float64{
		1, 2, 6, 8, 5, 6,
		14, 16, 36, 40, 22, 24,
		26, 28, 60, 64, 34, 36,
		19, 20, 42, 44, 23, 24,
	})
	if !mat.Equal(slipRet, tarSlipRet) {
		t.Fatalf("slip unpack not right need:\n%v\nbut:\n%v\n", tarSlipRet, slipRet)
	}
}

func TestConvPacker2(t *testing.T) {
	packer := NewConvPacker(
		[]int{4, 3, 2},
		[]int{2, 2},
		[]int{2, 2},
		true,
	)
	data := mat.NewVecDense(24, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
	slip := packer.Pack(data)
	tarSlip := mat.NewDense(4, 8, []float64{
		1, 2, 3, 4, 7, 8, 9, 10,
		5, 6, 0, 0, 11, 12, 0, 0,
		13, 14, 15, 16, 19, 20, 21, 22,
		17, 18, 0, 0, 23, 24, 0, 0,
	})
	if !mat.Equal(slip, tarSlip) {
		t.Fatalf("slip pack not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarSlip), mat.Formatted(slip))
	}
	slipRet := packer.UnPack(slip)
	if !mat.Equal(slipRet, data) {
		t.Fatalf("slip unpack not right need:\n%v\nbut:\n%v\n", data, slipRet)
	}
}

func TestMatColPicker1(t *testing.T) {
	picker := NewMatColPicker(
		[]int{2, 3, 2},
		2,
	)
	data := mat.NewDense(12, 4, []float64{
		10, 11, 12, 13,
		14, 15, 16, 17,
		18, 19, 20, 21,
		22, 23, 24, 25,
		26, 27, 28, 29,
		30, 31, 32, 33,
		34, 35, 36, 37,
		38, 39, 40, 41,
		42, 43, 45, 46,
		47, 48, 49, 50,
		51, 52, 53, 54,
		55, 56, 57, 58,
	})
	newData := picker.Pick(data)
	tarData := mat.NewDense(24, 2, []float64{
		10, 14,
		11, 15,
		12, 16,
		13, 17,
		18, 22,
		19, 23,
		20, 24,
		21, 25,
		26, 30,
		27, 31,
		28, 32,
		29, 33,
		34, 38,
		35, 39,
		36, 40,
		37, 41,
		42, 47,
		43, 48,
		45, 49,
		46, 50,
		51, 55,
		52, 56,
		53, 57,
		54, 58,
	})
	if !mat.Equal(tarData, newData) {
		t.Fatalf("pick not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarData), mat.Formatted(newData))
	}
	retData := picker.Pick(newData)
	if !mat.Equal(data, retData) {
		t.Fatalf("pick return not right need:\n%v\nbut:\n%v\n", mat.Formatted(data), mat.Formatted(retData))
	}
}

func TestMatColPicker2(t *testing.T) {
	picker := NewMatColPicker(
		[]int{2, 3, 2},
		1,
	)
	data := mat.NewDense(12, 4, []float64{
		10, 11, 12, 13,
		14, 15, 16, 17,
		18, 19, 20, 21,
		22, 23, 24, 25,
		26, 27, 28, 29,
		30, 31, 32, 33,
		34, 35, 36, 37,
		38, 39, 40, 41,
		42, 43, 45, 46,
		47, 48, 49, 50,
		51, 52, 53, 54,
		55, 56, 57, 58,
	})
	newData := picker.Pick(data)
	tarData := mat.NewDense(16, 3, []float64{
		10, 18, 26,
		14, 22, 30,
		11, 19, 27,
		15, 23, 31,
		12, 20, 28,
		16, 24, 32,
		13, 21, 29,
		17, 25, 33,
		34, 42, 51,
		38, 47, 55,
		35, 43, 52,
		39, 48, 56,
		36, 45, 53,
		40, 49, 57,
		37, 46, 54,
		41, 50, 58,
	})
	if !mat.Equal(tarData, newData) {
		t.Fatalf("pick not right need:\n%v\nbut:\n%v\n", mat.Formatted(tarData), mat.Formatted(newData))
	}
	retData := picker.Pick(newData)
	if !mat.Equal(data, retData) {
		t.Fatalf("pick return not right need:\n%v\nbut:\n%v\n", mat.Formatted(data), mat.Formatted(retData))
	}
}
