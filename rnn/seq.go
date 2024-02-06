package rnn

import "gonum.org/v1/gonum/mat"

type SeqWrapper struct {
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
