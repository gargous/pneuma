package nn

import (
	"gonum.org/v1/gonum/mat"
)

type OptNormal struct {
	lr float64
}

func NewOptNormal(lr float64) *OptNormal {
	return &OptNormal{
		lr: lr,
	}
}

func rangeOptimize(datas, deltas []mat.Matrix, vec func(i int, x, dx *mat.VecDense), dense func(i int, x, dx *mat.Dense)) {
	for i := 0; i < len(datas); i++ {
		x := datas[i]
		dx := deltas[i]
		switch rx := x.(type) {
		case *mat.VecDense:
			vec(i, rx, dx.(*mat.VecDense))
		case *mat.Dense:
			dense(i, rx, dx.(*mat.Dense))
		}
	}
}

func (opt *OptNormal) Copy(src *OptNormal) {
	opt.lr = src.lr
}

func (opt *OptNormal) Update(datas, deltas []mat.Matrix) {
	rangeOptimize(datas, deltas,
		func(i int, x, dx *mat.VecDense) {
			x.AddScaledVec(x, -opt.lr, dx)
		},
		func(i int, x, dx *mat.Dense) {
			xr, xc := x.Dims()
			v := mat.NewDense(xr, xc, nil)
			v.Scale(opt.lr, dx)
			x.Sub(x, v)
		})
}

type OptMomentum struct {
	lr float64
	v  []mat.Matrix
	mt float64
}

func NewOptMomentum(lr, mt float64) *OptMomentum {
	return &OptMomentum{
		lr: lr,
		mt: mt,
	}
}

func (opt *OptMomentum) init(datas, deltas []mat.Matrix) {
	if len(opt.v) == len(datas) {
		return
	}
	opt.v = make([]mat.Matrix, len(datas))
	rangeOptimize(datas, deltas,
		func(i int, x, dx *mat.VecDense) {
			opt.v[i] = mat.NewVecDense(x.Len(), nil)
		},
		func(i int, x, dx *mat.Dense) {
			r, c := x.Dims()
			opt.v[i] = mat.NewDense(r, c, nil)
		})
}

func (opt *OptMomentum) Copy(src *OptMomentum) {
	opt.lr = src.lr
	opt.mt = src.mt
	if len(src.v) != 0 {
		opt.v = make([]mat.Matrix, len(src.v))
		rangeOptimize(src.v, src.v,
			func(i int, x, dx *mat.VecDense) {
				opt.v[i] = mat.VecDenseCopyOf(x)
			}, func(i int, x, dx *mat.Dense) {
				opt.v[i] = mat.DenseCopyOf(x)
			})
	}
}

func (opt *OptMomentum) Update(datas, deltas []mat.Matrix) {
	opt.init(datas, deltas)
	rangeOptimize(datas, deltas,
		func(i int, x, dx *mat.VecDense) {
			v := opt.v[i].(*mat.VecDense)
			v.ScaleVec(opt.mt, v)
			v.AddScaledVec(v, -opt.lr, dx)
			x.AddVec(x, v)
		},
		func(i int, x, dx *mat.Dense) {
			v := opt.v[i].(*mat.Dense)
			v.Scale(opt.mt, v)
			xr, xc := x.Dims()
			newV := mat.NewDense(xr, xc, nil)
			newV.Scale(opt.lr, dx)
			v.Sub(v, newV)
			x.Add(x, v)
		})
}
