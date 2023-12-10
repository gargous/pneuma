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

func floorTen(v float64) float64 {
	ret := 1.0
	for {
		v /= 10
		ret *= 10
		if v < 1 {
			break
		}
	}
	return ret
}

func primeFac(v int) int {
	ret := 0
	for i := 2; i < v; {
		if v%i == 0 {
			ret++
			v /= i
		} else {
			i++
		}
	}
	return ret
}

func makePrimeSample(trainSamp, testSamp []nnSample, lables []*mat.VecDense, withFac bool) {
	prime := []int{2, 3, 5, 7, 11}
	noPrim := []int{4, 6, 8, 9, 10}
	noPrimFac := []int{}
	for _, v := range noPrim {
		noPrimFac = append(noPrimFac, primeFac(v))
	}
	scnt := len(trainSamp) + len(testSamp)
	m := scnt * 100
	grad := 100000.0
	for i := 12; i < m; i++ {
		sqrtn := int(math.Sqrt(float64(i)))
		primed := true
		for j := 0; j < len(prime); j++ {
			if prime[j] > sqrtn {
				break
			}
			if i%prime[j] == 0 {
				primed = false
				break
			}
		}
		if primed {
			lastPrime := prime[len(prime)-1]
			prime = append(prime, i)
			noPrimv := lastPrime + 1 + rand.Intn(i-lastPrime-1)
			noPrim = append(noPrim, noPrimv)
			if withFac {
				noPrimFac = append(noPrimFac, primeFac(noPrimv))
			}
		}
		if len(prime) > scnt && len(noPrim) > scnt {
			break
		}
	}
	minCnt := int(math.Min(float64(scnt), math.Min(float64(len(prime)), float64(len(noPrim)))))
	grad = floorTen(float64(prime[minCnt-1]))
	testidx := 0
	trainidx := 0
	for _, i := range rand.Perm(minCnt) {
		pring := floorTen(float64(prime[i])) / grad
		prinv := float64(prime[i]) / grad
		noPrinvg := floorTen(float64(noPrim[i])) / grad
		noPrinv := float64(noPrim[i]) / grad
		if withFac {
			gradF := floorTen(float64(noPrimFac[minCnt-1]))
			noPrinvf := float64(noPrimFac[i]) / gradF
			if testidx < len(testSamp) {
				testSamp[testidx] = nnSample{mat.NewVecDense(3, []float64{prinv, pring, 0}), lables[0]}
				testSamp[testidx+1] = nnSample{mat.NewVecDense(3, []float64{noPrinv, noPrinvg, noPrinvf}), lables[1]}
				testidx += 2
			} else {
				if trainidx < len(trainSamp) {
					trainSamp[trainidx] = nnSample{mat.NewVecDense(3, []float64{prinv, pring, 0}), lables[0]}
					trainSamp[trainidx+1] = nnSample{mat.NewVecDense(3, []float64{noPrinv, noPrinvg, noPrinvf}), lables[1]}
					trainidx += 2
				}
			}
		} else {
			if testidx < len(testSamp) {
				testSamp[testidx] = nnSample{mat.NewVecDense(2, []float64{prinv, pring}), lables[0]}
				testSamp[testidx+1] = nnSample{mat.NewVecDense(2, []float64{noPrinv, noPrinvg}), lables[1]}
				testidx += 2
			} else {
				if trainidx < len(trainSamp) {
					trainSamp[trainidx] = nnSample{mat.NewVecDense(2, []float64{prinv, pring}), lables[0]}
					trainSamp[trainidx+1] = nnSample{mat.NewVecDense(2, []float64{noPrinv, noPrinvg}), lables[1]}
					trainidx += 2
				}
			}
		}

	}
	fmt.Println("makePrimeSample done", len(prime), len(noPrim), grad)
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
