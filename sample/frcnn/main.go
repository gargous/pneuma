package main

import (
	"fmt"
	"net/http"
	"pneuma/cnn"
	"pneuma/common"
	"pneuma/cu"
	"pneuma/data"
	"pneuma/frcnn"
	"pneuma/nn"
	"pneuma/sample"

	_ "net/http/pprof"

	"gonum.org/v1/gonum/mat"
)

func voc() {
	epoch := 16
	batch := 1
	samplingRate := 100.0
	lineChartChild := 2
	batchNormMinSTD := 0.0001
	batchNormMT := 0.9
	learingRate := 0.001

	trainCnt := 12000
	testCnt := 1200
	valiCnt := 1200
	testCntPTime := 20
	valiCntPTime := 20
	loadCnt := 20
	size := []int{640, 640, 3}
	dataSet := newVOCSet(size, trainCnt, testCnt, valiCnt)

	eng := cu.NewEngine()
	b := frcnn.NewModelBuilder(size, []int{3, 3, 10})
	b.Cs(4, func(i int, ms *nn.ModelSample) {
		cal := cu.NewMatCaltor(eng)
		ms.Lay(cu.NewHLayerConv(cal, cnn.NewConvKParam([]int{3, 3, 16 * (i + 1)}, []int{1, 1}, cnn.ConvKernalPadAll)))
		ms.Opt(cu.NewOptNormal(cal, learingRate))
	})
	b.CLay(func() common.IHLayer {
		return cnn.NewHLayerConvBatchNorm(batchNormMinSTD, batchNormMT)
	})
	b.CLay(func() common.IHLayer {
		return nn.NewHLayerRelu()
	})
	b.CLay(func() common.IHLayer {
		return cnn.NewHLayerMaxPooling(cnn.NewConvKParam([]int{2, 2}, []int{2, 2}, cnn.ConvKernalPadFit))
	})
	b.RPN(func(score, trans cnn.ConvKernalParam) (scnv common.IHLayerSizeIniter, tcnv common.IHLayerSizeIniter, opt common.IOptimizer) {
		cal := cu.NewMatCaltor(eng)
		return cu.NewHLayerConv(cal, score), cu.NewHLayerConv(cal, trans), cu.NewOptNormal(cal, learingRate)
	})

	m := b.Build()
	lineChart := sample.NewLineChart("voc")
	lineChart.Reg("acc_vali", "acc_test", "loss_train", "loss_vali", "loss_test")
	fmt.Printf("train start\n")
	for e := 0; e < epoch; e++ {
		dataSet.Trains.ResetLoad()
		if e >= lineChartChild {
			makeSampleRecu(dataSet.Trains, trainCnt, loadCnt, batch, func(trainx, trainy []*mat.Dense) {
				m.Trains(trainx, trainy)
			})
		} else {
			trainTimes := 0
			sampFreq := int(float64(trainCnt/loadCnt) / samplingRate)
			makeSampleRecu(dataSet.Trains, trainCnt, loadCnt, batch, func(trainx, trainy []*mat.Dense) {
				m.Trains(trainx, trainy)
				if trainTimes%sampFreq == 0 {
					child := lineChart.Child(e)
					testModel(child, m, dataSet, loadCnt, batch, valiCntPTime, testCntPTime)
					fmt.Printf("test at:%d, trainTimes:%d, test info :%s\n", e, trainTimes, child.Format(child.Len()-1))
				}
				trainTimes++
			})
		}
		testModel(lineChart, m, dataSet, loadCnt, batch, valiCnt, testCnt)
		lineChart.Set(2, m.LossPopMean())
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	fmt.Printf("train end\n")
	lineChart.Draw()
}

func testModel(chart *sample.LineChart, m *frcnn.Model, dataSet *data.VOCSet, loadCnt, batch, valiCnt, testCnt int) {
	dataSet.Valids.ResetLoad()
	dataSet.Tests.ResetLoad()
	loss_vali, acc_vali := testRecu(m, dataSet.Valids, valiCnt, loadCnt, batch)
	loss_test, acc_test := testRecu(m, dataSet.Tests, testCnt, loadCnt, batch)
	chart.Append(acc_vali, acc_test, m.LossLatest(), loss_vali, loss_test)
}

func main() {
	go func() {
		fmt.Println(http.ListenAndServe(":6060", nil))
	}()
	voc()
	//testSample()
	//testAnchor()
}
