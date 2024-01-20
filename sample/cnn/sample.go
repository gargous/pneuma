package main

import (
	"image/jpeg"
	"io/fs"
	"os"
	"path/filepath"
	"pneuma/data"
	"pneuma/sample"
	"strconv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func makeHandWrittenSample(trainSamp, testSamp []sample.NNSample, labelCnt int) (size []int, labels []*mat.VecDense) {
	labels = make([]*mat.VecDense, labelCnt)
	for i := 0; i < len(labels); i++ {
		labels[i] = mat.NewVecDense(len(labels), nil)
		labels[i].SetVec(i, 1)
	}
	trainPath := filepath.Join("./resource", "handwritten", "img", "train")
	testPath := filepath.Join("./resource", "handwritten", "img", "test")
	trainFiles := make([][]string, len(labels))
	testFiles := make([][]string, len(labels))
	for i := 0; i < len(labels); i++ {
		trains, err := os.ReadDir(filepath.Join(trainPath, strconv.Itoa(i)))
		if err != nil {
			panic(err)
		}
		for _, v := range trains {
			trainFiles[i] = append(trainFiles[i], filepath.Join(trainPath, strconv.Itoa(i), v.Name()))
		}
		tests, err := os.ReadDir(filepath.Join(testPath, strconv.Itoa(i)))
		if err != nil {
			panic(err)
		}
		for _, v := range tests {
			testFiles[i] = append(testFiles[i], filepath.Join(testPath, strconv.Itoa(i), v.Name()))
		}
	}
	labIndx := 0
	for i := 0; i < len(trainSamp); i++ {
		var oneLabs []string
		for j := 0; j < len(labels); j++ {
			oneLabs = trainFiles[labIndx]
			if len(oneLabs) == 0 {
				labIndx = (labIndx + 1) % len(labels)
			} else {
				break
			}
		}
		if len(oneLabs) == 0 {
			panic("too much train sample")
		}
		oneLabFile := oneLabs[0]
		trainFiles[labIndx] = oneLabs[1:]
		file, err := os.Open(oneLabFile)
		if err != nil {
			panic(err)
		}
		img, err := jpeg.Decode(file)
		if err != nil {
			panic(err)
		}
		if len(size) == 0 {
			b := img.Bounds().Size()
			size = []int{b.Y, b.X, 1}
		}
		x := data.ImgToVecData(img, size)
		trainSamp[i] = sample.NNSample{mat.NewVecDense(len(x), x), labels[labIndx]}
		labIndx = (labIndx + 1) % len(labels)
	}
	labIndx = 0
	for i := 0; i < len(testSamp); i++ {
		var oneLabs []string
		for j := 0; j < len(labels); j++ {
			oneLabs = testFiles[labIndx]
			if len(oneLabs) == 0 {
				labIndx = (labIndx + 1) % len(labels)
			} else {
				break
			}
		}
		if len(oneLabs) == 0 {
			panic("too much test sample")
		}
		oneLabFile := oneLabs[0]
		testFiles[labIndx] = oneLabs[1:]
		file, err := os.Open(oneLabFile)
		if err != nil {
			panic(err)
		}
		img, err := jpeg.Decode(file)
		if err != nil {
			panic(err)
		}
		x := data.ImgToVecData(img, size)
		testSamp[i] = sample.NNSample{mat.NewVecDense(len(x), x), labels[labIndx]}
		labIndx = (labIndx + 1) % len(labels)
	}
	return
}

func testHandWrittenSample(size []int, trainSamps ...sample.NNSample) {
	samplePath := filepath.Join("./resource", "handwritten", "img", "samp")
	for i := 0; i < len(trainSamps); i++ {
		sampData := trainSamps[i].X
		sampLab := trainSamps[i].Y
		labIdx := floats.MaxIdx(sampLab.RawVector().Data)
		fPath := filepath.Join(samplePath, strconv.Itoa(labIdx))
		os.MkdirAll(fPath, fs.ModePerm)
		fname := strconv.Itoa(i) + ".jpg"
		file, err := os.Create(filepath.Join(fPath, fname))
		if err != nil {
			panic(err)
		}
		img := data.VecDataToImage(sampData.RawVector().Data, size)
		jpeg.Encode(file, img, &jpeg.Options{})
	}
}
