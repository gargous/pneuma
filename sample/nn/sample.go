package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type nnSample struct {
	x *mat.VecDense
	y *mat.VecDense
}

func makePrimeSample(trainSamp, testSamp []nnSample, lables []*mat.VecDense) {
	prime := []int{
		2, 3, 5, 7, 11, 13, 17, 19, 23,
	}
	noPrim := []int{
		1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22,
	}
	scnt := len(trainSamp) + len(testSamp)
	m := scnt * 100
	n := 29
	grad := 0.0
	for i := 9; i < m; i++ {
		sqrtn := int(math.Sqrt(float64(n)))
		j := 0
		primed := false
		for ; j < len(prime); j++ {
			if prime[j] > sqrtn {
				primed = true
				break
			}
		}
		if primed {
			lastPrime := prime[len(prime)-1]
			prime = append(prime, n)
			noPrim = append(noPrim, lastPrime+1+rand.Intn(n-lastPrime-1))
		}
		n += 2
		grad = float64(n)
		if len(prime) > scnt && len(noPrim) > scnt {
			break
		}
	}
	minCnt := int(math.Min(float64(len(prime)), float64(len(noPrim))))
	testidx := 0
	trainidx := 0
	for _, i := range rand.Perm(minCnt) {
		prinv := float64(prime[i]) / grad
		noPrinv := float64(noPrim[i]) / grad
		if testidx < len(testSamp) {
			testSamp[testidx] = nnSample{mat.NewVecDense(1, []float64{prinv}), lables[0]}
			testSamp[testidx+1] = nnSample{mat.NewVecDense(1, []float64{noPrinv}), lables[1]}
			testidx += 2
		} else {
			if trainidx < len(trainSamp) {
				trainSamp[trainidx] = nnSample{mat.NewVecDense(1, []float64{prinv}), lables[0]}
				trainSamp[trainidx+1] = nnSample{mat.NewVecDense(1, []float64{noPrinv}), lables[1]}
				trainidx += 2
			}
		}
	}
	fmt.Println("makePrimeSample done", n, len(prime), len(noPrim))
}

func makeBTZeroSample(trainSamp, testSamp []nnSample, lables []*mat.VecDense) {
	for i := 0; i < len(trainSamp); i += 2 {
		trainSamp[i] = nnSample{mat.NewVecDense(1, []float64{rand.Float64()}), lables[0]}
		trainSamp[i+1] = nnSample{mat.NewVecDense(1, []float64{-rand.Float64()}), lables[1]}
		if i >= len(testSamp) {
			continue
		}
		testSamp[i] = nnSample{mat.NewVecDense(1, []float64{rand.Float64()}), lables[0]}
		testSamp[i+1] = nnSample{mat.NewVecDense(1, []float64{-rand.Float64()}), lables[1]}
	}
}

func makeRGBASample(trainSamp, testSamp []nnSample, lables []*mat.VecDense) {
	for i := 0; i < len(trainSamp); i += 2 {
		trainSamp[i] = nnSample{mat.NewVecDense(4, []float64{
			(225 + 30*rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(245 + 10*rand.Float64()) / 255.0,
		}), lables[0]}
		trainSamp[i+1] = nnSample{mat.NewVecDense(4, []float64{
			(225 * rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(245 * rand.Float64()) / 255.0,
		}), lables[1]}
		if i >= len(testSamp) {
			continue
		}
		testSamp[i] = nnSample{mat.NewVecDense(4, []float64{
			(225 + 30*rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(245 + 10*rand.Float64()) / 255.0,
		}), lables[0]}
		testSamp[i+1] = nnSample{mat.NewVecDense(4, []float64{
			(225 * rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(245 * rand.Float64()) / 255.0,
		}), lables[1]}
	}
}
