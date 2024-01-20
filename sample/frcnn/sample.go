package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"pneuma/data"
	"pneuma/frcnn"
	"pneuma/sample"

	"gonum.org/v1/gonum/mat"
)

func newVOCSet(size []int, trainCnt, testCnt, valiCnt int) (vocSet *data.VOCSet) {
	fmt.Println("new VOCSet start")
	vocSet = data.NewVOCSet(size, filepath.Join("./resource", "voc"))
	err := vocSet.ReadAnnot()
	if err != nil {
		panic(err)
	}
	vocSet.Trains.Reindex(rand.Perm(trainCnt))
	vocSet.Tests.Reindex(rand.Perm(testCnt))
	vocSet.Valids.Reindex(rand.Perm(valiCnt))
	fmt.Println("new VOCSet end")
	return
}

func makeSample(set *data.VOCDatas, cnt, batch int) ([]*mat.Dense, []*mat.Dense) {
	samples := make([]sample.NNSample, cnt)
	datas, err := set.Load(cnt)
	if err != nil {
		panic(err)
	}
	for i := 0; i < cnt; i++ {
		img := datas[i].Img
		samples[i].X = mat.NewVecDense(len(img), img)
		bnd := set.DataToBnd(datas[i])
		samples[i].Y = mat.NewVecDense(len(bnd), bnd)
	}
	x, y := sample.StackSample(samples, batch)
	fmt.Println("make sample", set.IdxPath, cnt)
	return x, y
}

func makeSampleRecu(set *data.VOCDatas, allCnt, loadCnt, batch int, cb func(x, y []*mat.Dense)) {
	cnt := allCnt / loadCnt
	for i := 0; i < cnt; i++ {
		cb(makeSample(set, loadCnt, batch))
	}
}

func testRecu(m *frcnn.Model, set *data.VOCDatas, allCnt, loadCnt, batch int) (loss, acc float64) {
	cnt := 0.0
	makeSampleRecu(set, allCnt, loadCnt, batch, func(valix, valiy []*mat.Dense) {
		_loss, _acc := m.Tests(valix, valiy)
		acc += _acc
		loss += _loss
		cnt += 1
	})
	return loss / cnt, acc / cnt
}

func testSample() {
	trainCnt := 12000
	testCnt := 1200
	valiCnt := 1200
	size := []int{640, 640, 3}
	dataSet := newVOCSet(size, trainCnt, testCnt, valiCnt)
	trainDatas, err := dataSet.Trains.Pop(3)
	if err != nil {
		panic(err)
	}
	fPath := filepath.Join(dataSet.RootPath, "TestSample")
	for _, d := range trainDatas {
		os.MkdirAll(fPath, fs.ModePerm)
		f, err := os.Create(filepath.Join(fPath, d.ImgName))
		if err != nil {
			panic(err)
		}
		err = d.BndImg(f, size)
		if err != nil {
			panic(err)
		}
		f.Close()
	}
}

func testAnchor() {
	size := []int{640, 640, 3}
	param := frcnn.NewRPNParam(size, []int{3, 3, 10})
	rpn := frcnn.NewRPN(param)
	rpn.InitSize([]int{20, 20, 100})
	img := image.NewRGBA(image.Rect(0, 0, size[1], size[0]))
	white := image.NewUniform(color.RGBA{255, 255, 255, 255})
	draw.Draw(img, img.Rect, white, image.Point{}, draw.Src)
	bnds := rpn.PropBounds.ToDense()
	fPath := filepath.Join("./resource", "voc", "TestSample")
	os.MkdirAll(fPath, fs.ModePerm)
	f, err := os.Create(filepath.Join(fPath, "propAnchor.jpg"))
	if err != nil {
		panic(err)
	}
	for i := 0; i < bnds.RawMatrix().Rows; i++ {
		if rpn.PropInners[i] {
			data.DrawBnd(img, bnds.RawRowView(i), 1)
		}
	}
	err = jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	if err != nil {
		panic(err)
	}
	f.Close()
}
