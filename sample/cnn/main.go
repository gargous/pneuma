package main

import (
	"fmt"
	"pneuma/cnn"
	"pneuma/common"
	"pneuma/cu"
	"pneuma/nn"
	"pneuma/sample"
)

func handwritten() {
	epoch := 16
	batch := 16
	samplingRate := 100.0
	lineChartChild := 2
	batchNormMinSTD := 0.0001
	batchNormMT := 0.9
	learingRate := 0.001
	optMT := 0.1
	trainSamp := make([]sample.NNSample, 50000)
	testSamp := make([]sample.NNSample, 1000)
	size, labels := makeHandWrittenSample(trainSamp, testSamp, 10)
	trainSamp, testSamp, valiSamp := sample.RandSample(trainSamp, testSamp)
	trainx, trainy := sample.StackSample(trainSamp, batch)
	testx, testy := sample.StackSample(testSamp, batch)
	valix, valiy := sample.StackSample(valiSamp, batch)

	b := cnn.NewModelBuilder(size)
	b.Conv([]int{5, 5, 10}, []int{1, 1}, true)
	b.Conv([]int{3, 3, 20}, []int{1, 1}, true)
	b.Conv([]int{2, 2, 40}, []int{1, 1}, true)
	b.FSize(len(labels))
	b.CLayer(func(inpSize []int) common.IHLayer {
		return cnn.NewHLayerConvBatchNorm(inpSize, batchNormMinSTD, batchNormMT)
	})
	b.CLayer(func(inpSize []int) common.IHLayer { return nn.NewHLayerRelu() })
	b.CLayer(func(inpSize []int) common.IHLayer {
		return cnn.NewHLayerMaxPooling(inpSize, []int{2, 2}, []int{2, 2}, true)
	})
	b.FLayer(func() common.IHLayer { return nn.NewHLayerBatchNorm(batchNormMinSTD, batchNormMT) })
	b.FLayer(func() common.IHLayer { return nn.NewHLayerRelu() })
	b.Optimizer(func() common.IOptimizer { return nn.NewOptMomentum(learingRate, optMT) })
	//b.Optimizer(func() common.IOptimizer { return nn.NewOptNormal(learingRate) })
	b.Target(func() common.ITarget { return nn.NewTarCE() })
	m := b.Build()

	eng := cu.NewEngine()
	for i := 0; i < m.LayerCnt(); i++ {
		_, hls := m.Layer(i)
		if conv, isConv := hls[0].(*cnn.HLayerConv); isConv {
			conv.SetCaltor(cu.NewConvCalter(eng))
		}
	}

	lineChart := sample.NewLineChart("handwritten")
	lineChart.Reg("acc_vali", "acc_test", "loss_train", "loss_vali", "loss_test")
	fmt.Printf("train start\n")
	for e := 0; e < epoch; e++ {
		if e >= lineChartChild {
			m.Trains(trainx, trainy)
		} else {
			sampFreq := int(float64(len(trainx)) / samplingRate)
			m.TrainTimes(trainx, trainy, func(trainTimes int) {
				if trainTimes%sampFreq == 0 {
					child := lineChart.Child(e)
					vpred := m.Predicts(valix)
					tpred := m.Predicts(testx)
					child.Append(m.Accs(vpred, valiy), m.Accs(tpred, testy), m.LossLatest(), m.MeanLosses(vpred, valiy), m.MeanLosses(tpred, testy))
					fmt.Printf("train at:%d, trainTimes:%d, test info :%s\n", e, trainTimes, child.Format(child.Len()-1))
				}
			})
		}
		vpred := m.Predicts(valix)
		tpred := m.Predicts(testx)
		lineChart.Append(m.Accs(vpred, valiy), m.Accs(tpred, testy), m.LossPopMean(), m.MeanLosses(vpred, valiy), m.MeanLosses(tpred, testy))
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	fmt.Printf("train end\n")
	lineChart.Draw()
}

func main() {
	handwritten()
}
