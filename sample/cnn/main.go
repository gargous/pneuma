package main

import (
	"fmt"
	"pneuma/cnn"
	"pneuma/common"
	"pneuma/nn"
	"pneuma/sample"
)

func handwritten() {
	epoch := 16
	batch := 8
	samplingRate := 100.0
	lineChartChild := 2
	batchNormMinSTD := 0.0001
	batchNormMT := 0.9
	learingRate := 0.001
	optMT := 0.1
	trainSamp := make([]sample.NNSample, 50000)
	testSamp := make([]sample.NNSample, 1000)
	size, labels := makeHandWrittenSample(trainSamp, testSamp, 10)
	trainSamp, testx, testy, valix, valiy := sample.RandSample(trainSamp, testSamp)
	trainx, trainy := sample.StackSample(trainSamp, batch)

	b := cnn.NewModelBuilder(size)
	b.Conv([]int{2, 2, 8}, []int{2, 2}, true)
	b.Conv([]int{2, 2, 16}, []int{2, 2}, true)
	b.Conv([]int{2, 2, 32}, []int{2, 2}, true)
	b.FSize(len(labels))
	//b.CLayer(func(inpSize []int) common.IHLayer { return nn.NewHLayerBatchNorm(batchNormMinSTD, batchNormMT) })
	b.CLayer(func(inpSize []int) common.IHLayer { return nn.NewHLayerRelu() })
	//b.CLayer(func(inpSize []int) common.IHLayer {
	//	return cnn.NewHLayerMaxPooling(inpSize, []int{2, 2}, []int{2, 2}, true)
	//})
	b.FLayer(func() common.IHLayer { return nn.NewHLayerBatchNorm(batchNormMinSTD, batchNormMT) })
	b.FLayer(func() common.IHLayer { return nn.NewHLayerRelu() })
	b.Optimizer(func() common.IOptimizer { return nn.NewOptMomentum(learingRate, optMT) })
	//b.Optimizer(func() common.IOptimizer { return nn.NewOptNormal(learingRate) })
	b.Target(func() common.ITarget { return nn.NewTarCE() })
	m := b.Build()

	lineChart := sample.NewLineChart("handwritten")
	lineChart.Reg("acc_vali", "acc_test", "loss_train", "loss_vali", "loss_test")
	fmt.Printf("train start\n")
	for e := 0; e < epoch; e++ {
		if e >= lineChartChild {
			m.TrainEpoch(trainx, trainy)
		} else {
			sampFreq := int(float64(len(trainx)) / samplingRate)
			m.TrainEpochTimes(trainx, trainy, func(trainTimes int) {
				if trainTimes%sampFreq == 0 {
					child := lineChart.Child(e)
					vpred := m.Predict(valix)
					tpred := m.Predict(testx)
					child.Append(m.Acc(vpred, valiy), m.Acc(tpred, testy), m.LossLatest(), m.Loss(vpred, valiy), m.Loss(tpred, testy))
					fmt.Printf("train at:%d, trainTimes:%d, test info :%s\n", e, trainTimes, child.Format(child.Len()-1))
				}
			})
		}
		vpred := m.Predict(valix)
		tpred := m.Predict(testx)
		lineChart.Append(m.Acc(vpred, valiy), m.Acc(tpred, testy), m.LossPopMean(), m.Loss(vpred, valiy), m.Loss(tpred, testy))
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	fmt.Printf("train end\n")
	lineChart.Draw()
}

func main() {
	handwritten()
}
