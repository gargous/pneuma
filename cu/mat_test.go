package cu

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDenseCaltorCU1(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.Mul(dst, a, b, false, false)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU2(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.Mul(dst, a, b, false, true)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU3(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(a, b)
	c.Add(a, b)
	c.End(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU4(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(a, b)
	c.AddSlice(a, b, 0, 0, 3)
	c.End(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU5(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(a, b)
	c.AddSlice(a, b, 3, 0, 3)
	c.End(a)
	if !mat.Equal(a, tar) {
		t.Fatalf("mat Add errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(a))
	}
	e.Close()
}

func TestDenseCaltorCU6(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.Mul(dst, a, b, true, false)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU7(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, false, false, 2)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU8(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, false, false, 5)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU9(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, true, false, 2)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU10(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, true, false, 5)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU11(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, false, true, 2)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}

func TestDenseCaltorCU12(t *testing.T) {
	e := NewEngine()
	c := NewDenseCaltorCU(e)
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
	c.Start(dst, a, b)
	c.MulKSpan(dst, a, b, false, true, 5)
	c.End(dst)
	if !mat.Equal(dst, tar) {
		t.Fatalf("mat Mul errorneed:\n%v\nbut:\n%v\n", mat.Formatted(tar), mat.Formatted(dst))
	}
	e.Close()
}
