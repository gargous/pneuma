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
	m := (len(trainSamp) + len(testSamp)) * 2
	n := 29
	for i := 9; i < m; i++ {
		sqrtn := int(math.Sqrt(float64(n)))
		j := 0
		for ; j < len(prime); j++ {
			if prime[j] > sqrtn {
				prime = append(prime, n)
				break
			}
			if n%int(prime[j]) == 0 {
				noPrim = append(noPrim, n)
				break
			}
		}
		n += 2
	}
	rate := float64(len(trainSamp)) / float64(len(trainSamp)+len(testSamp))
	pri := 0
	upri := 0
	tri := 0
	tei := 0
	grad := 1000000.0
	for ; pri < int(rate*float64(len(prime))); pri++ {
		if len(trainSamp)/2 <= tri {
			break
		}
		trainSamp[tri] = nnSample{mat.NewVecDense(1, []float64{float64(prime[pri]) / grad}), lables[0]}
		tri++
	}
	for ; pri < len(prime); pri++ {
		if len(testSamp)/2 <= tei {
			break
		}
		testSamp[tei] = nnSample{mat.NewVecDense(1, []float64{float64(prime[pri]) / grad}), lables[0]}
		tei++
	}
	for i := tri; i < len(trainSamp); i++ {
		trainSamp[i] = nnSample{mat.NewVecDense(1, []float64{float64(noPrim[upri]) / grad}), lables[1]}
		upri++

	}
	for i := tei; i < len(testSamp); i++ {
		testSamp[i] = nnSample{mat.NewVecDense(1, []float64{float64(noPrim[upri]) / grad}), lables[1]}
		upri++
	}
	fmt.Println("makePrimeSample done", n)
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
