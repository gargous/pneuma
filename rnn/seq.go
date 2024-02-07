package rnn

import (
	"gonum.org/v1/gonum/mat"
)

func SeqN2N(x, y []*mat.Dense, cb func(*mat.Dense) *mat.Dense) {
	for i := 0; i < len(x); i++ {
		y[i] = cb(x[i])
	}
}

func SeqN2One(x, y []*mat.Dense, cb func(*mat.Dense) *mat.Dense) {
	for i := 0; i < len(x)-1; i++ {
		cb(x[i])
	}
	y[0] = cb(x[len(x)-1])
}

func SeqOne2N(x, y []*mat.Dense, cb func(*mat.Dense) *mat.Dense) {
	a := x[0]
	for i := 0; i < len(y); i++ {
		a = cb(a)
		y[i] = a
	}
}

func SumDense(dense []*mat.Dense) (ret *mat.Dense) {
	for _, one := range dense {
		if ret == nil {
			ret = mat.DenseCopyOf(one)
		} else {
			ret.Add(ret, one)
		}
	}
	return
}

func SumVecDense(vec []*mat.VecDense) (ret *mat.VecDense) {
	for _, one := range vec {
		if ret == nil {
			ret = mat.VecDenseCopyOf(one)
		} else {
			ret.AddVec(ret, one)
		}
	}
	return
}
