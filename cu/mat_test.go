package cu

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDenseCaltorCU1(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.Mul(dst, a, b, false, false)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU2(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.Mul(dst, a, b, false, true)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU3(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(2, 3, []float64{
		2, 5, 8,
		6, 9, 12,
	})
	c.CopyTo(a, b)
	c.Add(a, b)
	c.CopyBack(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU4(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(1, 3, []float64{
		1, 3, 5,
	})
	tar := mat.NewDense(2, 3, []float64{
		2, 5, 8,
		4, 5, 6,
	})
	c.CopyTo(a, b)
	c.AddSlice(a, b, 0, 0, 3)
	c.CopyBack(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU5(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(1, 3, []float64{
		1, 3, 5,
	})
	tar := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		5, 8, 11,
	})
	c.CopyTo(a, b)
	c.AddSlice(a, b, 3, 0, 3)
	c.CopyBack(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU6(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(3, 3, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(3, 3, []float64{
		9, 19, 29,
		12, 26, 40,
		15, 33, 51,
	})
	c.CopyTo(dst, a, b)
	c.Mul(dst, a, b, true, false)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU7(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, false, false, 2)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU8(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, false, false, 5)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU9(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(3, 3, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(3, 3, []float64{
		9, 19, 29,
		12, 26, 40,
		15, 33, 51,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, true, false, 2)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU10(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(3, 3, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(3, 3, []float64{
		9, 19, 29,
		12, 26, 40,
		15, 33, 51,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, true, false, 5)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU11(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, false, true, 2)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU12(t *testing.T) {
	e := NewEngine()
	c := NewMatCaltor(e)
	dst := mat.NewDense(2, 2, nil)
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 3, []float64{
		1, 3, 5,
		2, 4, 6,
	})
	tar := mat.NewDense(2, 2, []float64{
		22, 28,
		49, 64,
	})
	c.CopyTo(dst, a, b)
	c.MulKSpan(dst, a, b, false, true, 5)
	c.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU14(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	r, c := 3, 4
	a := mat.NewDense(r, c, []float64{
		1, 2, 3, 1,
		1, 2, 3, 2,
		1, 2, 3, 3,
	})
	b := mat.NewVecDense(r, []float64{
		1,
		2,
		3,
	})
	tar := mat.NewDense(r, c, []float64{
		2, 3, 4, 2,
		3, 4, 5, 4,
		4, 5, 6, 6,
	})
	cal.CopyTo(a, b)
	cal.AddScaledColByOne(a, 1, b)
	cal.CopyBack(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU15(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	r, c := 3, 4
	a := mat.NewDense(r, c, []float64{
		1, 2, 3, 1,
		1, 2, 3, 2,
		1, 2, 3, 3,
	})
	b := mat.NewVecDense(r, nil)
	tar := mat.NewVecDense(r, []float64{
		7,
		8,
		9,
	})
	cal.CopyTo(a, b)
	cal.AddScaledOneByCol(b, 1, a)
	cal.CopyBack(b)
	if !mat.Equal(b, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(b))
	}
	e.Close()
}

func TestDenseCaltorCU16(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	r, c := 4, 4
	a := mat.NewDense(r, c, []float64{
		1, 2, 3, 1,
		1, 2, 3, 2,
		1, 2, 3, 3,
		1, 2, 3, 4,
	})
	b := mat.NewVecDense(c*2, []float64{
		1, 2, 3, 4,
		2, 4, 6, 8,
	})
	tar := mat.NewDense(r, c, []float64{
		2, 4, 6, 5,
		3, 6, 9, 10,
		2, 4, 6, 7,
		3, 6, 9, 12,
	})
	cal.CopyTo(a, b)
	cal.AddScaledRowByOne(a, 1, b)
	cal.CopyBack(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU17(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	r, c := 4, 4
	a := mat.NewDense(r, c, []float64{
		1, 2, 3, 1,
		1, 2, 3, 2,
		1, 2, 3, 3,
		1, 2, 3, 4,
	})
	b := mat.NewVecDense(c*2, nil)
	tar := mat.NewVecDense(c*2, []float64{
		2, 4, 6, 4,
		2, 4, 6, 6,
	})
	cal.CopyTo(a, b)
	cal.AddScaledOneByRow(b, 1, a)
	cal.CopyBack(b)
	if !mat.Equal(b, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(b))
	}
	e.Close()
}
func TestDenseCaltorCU20(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	a := mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	ret := mat.NewVecDense(1, nil)
	tar := mat.NewVecDense(1, []float64{2})
	cal.CopyTo(a)
	fun := func() error {
		fa := cal.DeviceData(a)
		ret.SetVec(0, cal.e.Dnrm2(4, fa, 1))
		return cal.e.Err()
	}
	err := e.Do(fun)
	if err != nil {
		fmt.Println(err)
	}
	//cal.CopyBack(ret)
	if !mat.Equal(ret, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(ret))
	}
	b := mat.NewVecDense(4, []float64{
		2, 2, 2, 2,
	})
	ret2 := mat.NewVecDense(1, nil)
	tar = mat.NewVecDense(1, []float64{4})
	cal.CopyTo(b, ret2)
	fun = func() error {
		fa := cal.DeviceData(b)
		ret2.SetVec(0, cal.e.Dnrm2(4, fa, 1))
		return cal.e.Err()
	}
	e.Do(fun)
	//cal.CopyBack(ret2)
	if !mat.Equal(ret2, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(ret2))
	}
	e.Close()
}

func TestDenseCaltorCU21(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	a := mat.NewDense(3, 4, []float64{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
	})
	tar := mat.NewVecDense(3, []float64{
		2,
		4,
		6,
	})
	dst := mat.NewVecDense(3, []float64{
		0,
		0,
		6,
	})
	cal.CopyTo(dst, a)
	cal.NormSliceInc(dst, a, 0, 0, 1, 4)
	cal.NormSliceInc(dst, a, 1, 4, 1, 4)
	cal.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU22(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	a := mat.NewDense(3, 4, []float64{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
	})
	tar := mat.NewVecDense(3, []float64{
		2,
		4,
		6,
	})
	dst := mat.NewVecDense(3, nil)
	cal.CopyTo(dst, a)
	cal.NormOneByRow(dst, a)
	cal.CopyBack(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU30(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	a := mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	b := mat.NewDense(4, 2, []float64{
		1, 2,
		1, 2,
		1, 2,
		2, 3,
	})
	tar := mat.NewDense(4, 2, []float64{
		1, 2,
		2, 4,
		3, 6,
		8, 12,
	})
	cal.CopyTo(b)
	cal.MulElemColByOneHost(b, a)
	cal.CopyBack(b)
	if !mat.Equal(b, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(b))
	}
	e.Close()
}

func TestDenseCaltorCU40(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	dst := mat.NewVecDense(4, nil)
	a := mat.NewDense(4, 2, []float64{
		1, 2,
		2, 4,
		3, 6,
		8, 12,
	})
	b := mat.NewDense(4, 2, []float64{
		1, 2,
		1, 2,
		1, 2,
		2, 3,
	})
	tar := mat.NewVecDense(4, []float64{
		5,
		10,
		15,
		52,
	})
	cal.CopyTo(a, b)
	cal.DotRowByRowToHost(dst, a, b)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU41(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	dst := mat.NewVecDense(1, nil)
	a := mat.NewVecDense(2, []float64{
		1, 2,
	})
	b := mat.NewVecDense(2, []float64{
		1, 2,
	})
	tar := mat.NewVecDense(1, []float64{
		5,
	})
	cal.CopyTo(a, b)
	dst.SetVec(0, cal.DotSliceInc(a, b, 0, 0, 1, 0, 1, 2))
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU42(t *testing.T) {
	e := NewEngine()
	cal := NewMatCaltor(e)
	dst := 0.0
	a := mat.NewVecDense(2, []float64{
		1, 2,
	})
	b := mat.NewVecDense(2, []float64{
		1, 2,
	})
	tar := 5.0
	cal.CopyTo(a, b)
	fun := func() error {
		fa := cal.DeviceData(a)
		fb := cal.DeviceData(b)
		dst = cal.e.Ddot(2, fa, 1, fb, 1)
		return cal.e.Err()
	}
	err := cal.e.Do(fun)
	if err != nil {
		panic(err)
	}
	if tar != dst {
		t.Fatalf("mat Mul error need:\n%v\nbut:\n%v\n", tar, dst)
	}
	e.Close()
}
