package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"pneuma/sample"
	"strings"

	"gonum.org/v1/gonum/mat"
)

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

func makePrimeSample(trainSamp, testSamp []sample.NNSample, lables []*mat.VecDense, withFac bool) {
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
				testSamp[testidx] = sample.NNSample{mat.NewVecDense(3, []float64{prinv, pring, 0}), lables[0]}
				testSamp[testidx+1] = sample.NNSample{mat.NewVecDense(3, []float64{noPrinv, noPrinvg, noPrinvf}), lables[1]}
				testidx += 2
			} else {
				if trainidx < len(trainSamp) {
					trainSamp[trainidx] = sample.NNSample{mat.NewVecDense(3, []float64{prinv, pring, 0}), lables[0]}
					trainSamp[trainidx+1] = sample.NNSample{mat.NewVecDense(3, []float64{noPrinv, noPrinvg, noPrinvf}), lables[1]}
					trainidx += 2
				}
			}
		} else {
			if testidx < len(testSamp) {
				testSamp[testidx] = sample.NNSample{mat.NewVecDense(2, []float64{prinv, pring}), lables[0]}
				testSamp[testidx+1] = sample.NNSample{mat.NewVecDense(2, []float64{noPrinv, noPrinvg}), lables[1]}
				testidx += 2
			} else {
				if trainidx < len(trainSamp) {
					trainSamp[trainidx] = sample.NNSample{mat.NewVecDense(2, []float64{prinv, pring}), lables[0]}
					trainSamp[trainidx+1] = sample.NNSample{mat.NewVecDense(2, []float64{noPrinv, noPrinvg}), lables[1]}
					trainidx += 2
				}
			}
		}

	}
	fmt.Println("makePrimeSample done", len(prime), len(noPrim), grad)
}

func makeBTZeroSample(trainSamp, testSamp []sample.NNSample, lables []*mat.VecDense) {
	for i := 0; i < len(trainSamp); i += 2 {
		trainSamp[i] = sample.NNSample{mat.NewVecDense(1, []float64{rand.Float64()}), lables[0]}
		trainSamp[i+1] = sample.NNSample{mat.NewVecDense(1, []float64{-rand.Float64()}), lables[1]}
	}
	for i := 0; i < len(testSamp); i += 2 {
		testSamp[i] = sample.NNSample{mat.NewVecDense(1, []float64{rand.Float64()}), lables[0]}
		testSamp[i+1] = sample.NNSample{mat.NewVecDense(1, []float64{-rand.Float64()}), lables[1]}
	}
}

func makeBT5Sample(trainSamp, testSamp []sample.NNSample, lables []*mat.VecDense) {
	for i := 0; i < len(trainSamp); i += 2 {
		trainSamp[i] = sample.NNSample{mat.NewVecDense(1, []float64{0.5 * rand.Float64()}), lables[0]}
		trainSamp[i+1] = sample.NNSample{mat.NewVecDense(1, []float64{0.5 + 0.5*rand.Float64()}), lables[1]}
	}
	for i := 0; i < len(testSamp); i += 2 {
		testSamp[i] = sample.NNSample{mat.NewVecDense(1, []float64{0.5 * rand.Float64()}), lables[0]}
		testSamp[i+1] = sample.NNSample{mat.NewVecDense(1, []float64{0.5 + 0.5*rand.Float64()}), lables[1]}
	}
}

func makeRGBASample(trainSamp, testSamp []sample.NNSample, lables []*mat.VecDense) {
	for i := 0; i < len(trainSamp); i += 2 {
		trainSamp[i] = sample.NNSample{mat.NewVecDense(4, []float64{
			(225 + 30*rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(245 + 10*rand.Float64()) / 255.0,
		}), lables[0]}
		trainSamp[i+1] = sample.NNSample{mat.NewVecDense(4, []float64{
			(225 * rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(245 * rand.Float64()) / 255.0,
		}), lables[1]}
	}
	for i := 0; i < len(testSamp); i += 2 {
		testSamp[i] = sample.NNSample{mat.NewVecDense(4, []float64{
			(225 + 30*rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(30 * rand.Float64()) / 255.0,
			(245 + 10*rand.Float64()) / 255.0,
		}), lables[0]}
		testSamp[i+1] = sample.NNSample{mat.NewVecDense(4, []float64{
			(225 * rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(30 + 225*rand.Float64()) / 255.0,
			(245 * rand.Float64()) / 255.0,
		}), lables[1]}
	}
}

func readKRKData(fname string, lables []*mat.VecDense) ([]sample.NNSample, error) {
	fi, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer fi.Close()
	br := bufio.NewReader(fi)
	var datas []sample.NNSample
	for {
		line, _, err := br.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		oneData := strings.Split(string(line), ",")
		data := sample.NNSample{}
		if string(oneData[6]) == "draw" {
			data.Y = lables[0]
		} else {
			data.Y = lables[1]
		}
		var datax []float64
		for i := 0; i < 6; i++ {
			oneXData := int(rune(oneData[i][0]))
			if i%2 == 0 {
				oneXData -= 96
			} else {
				oneXData -= 48
			}
			datax = append(datax, float64(oneXData))
		}
		data.X = mat.NewVecDense(len(datax), datax)
		datas = append(datas, data)
	}

	return datas, nil
}

func makeKRKSample(trainSamp, testSamp []sample.NNSample, lables []*mat.VecDense) {
	samples, err := readKRKData(filepath.Join("resource", "krkopt.data"), lables)
	if err != nil {
		panic(err)
	}
	if len(samples) < len(trainSamp)+len(testSamp) {
		panic("too much sample")
	}
	idx := 0
	for i := 0; i < len(trainSamp); i++ {
		trainSamp[i] = samples[idx]
		idx++
	}
	for i := 0; i < len(testSamp); i++ {
		testSamp[i] = samples[idx]
		idx++
	}
}
