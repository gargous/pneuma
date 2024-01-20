package main

import (
	"fmt"
	"net/http"
	_ "net/http/pprof"
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
	//optMT := 0.1
	trainSamp := make([]sample.NNSample, 50000)
	testSamp := make([]sample.NNSample, 1000)
	size, labels := makeHandWrittenSample(trainSamp, testSamp, 10)
	trainSamp, testSamp, valiSamp := sample.RandSample(trainSamp, testSamp)
	trainx, trainy := sample.StackSample(trainSamp, batch)
	testx, testy := sample.StackSample(testSamp, batch)
	valix, valiy := sample.StackSample(valiSamp, batch)

	eng := cu.NewEngine()
	b := cnn.NewModelBuilder(size, []int{len(labels)})

	/*
		b.C().Build(func(mmb *cnn.ModelMiniBuilder) {
			mmb.Lay(cnn.NewHLayerConv(cnn.NewConvKParam([]int{5, 5, 10}, []int{1, 1}, true0)))
		})
		b.C().Build(func(mmb *cnn.ModelMiniBuilder) {
			mmb.Lay(cnn.NewHLayerConv(cnn.NewConvKParam([]int{3, 3, 20}, []int{1, 1}, true)))
		})
		b.C().Build(func(mmb *cnn.ModelMiniBuilder) {
			mmb.Lay(cnn.NewHLayerConv(cnn.NewConvKParam([]int{2, 2, 40}, []int{1, 1}, true)))
		})
		b.COpt(func() common.IOptimizer {
			return nn.NewOptNormal(learingRate)
		})
	*/

	b.C(func(mmb *nn.ModelSample) {
		cal := cu.NewMatCaltor(eng)
		mmb.Lay(cu.NewHLayerConv(cal, cnn.NewConvKParam([]int{5, 5, 10}, []int{1, 1}, cnn.ConvKernalPadFit)))
		mmb.Opt(cu.NewOptNormal(cal, learingRate))
	})
	b.C(func(mmb *nn.ModelSample) {
		cal := cu.NewMatCaltor(eng)
		mmb.Lay(cu.NewHLayerConv(cal, cnn.NewConvKParam([]int{3, 3, 20}, []int{1, 1}, cnn.ConvKernalPadFit)))
		mmb.Opt(cu.NewOptNormal(cal, learingRate))
	})
	b.C(func(mmb *nn.ModelSample) {
		cal := cu.NewMatCaltor(eng)
		mmb.Lay(cu.NewHLayerConv(cal, cnn.NewConvKParam([]int{2, 2, 40}, []int{1, 1}, cnn.ConvKernalPadFit)))
		mmb.Opt(cu.NewOptNormal(cal, learingRate))
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
	b.FLay(func() common.IHLayer { return nn.NewHLayerLinear() })
	b.FLay(func() common.IHLayer { return nn.NewHLayerBatchNorm(batchNormMinSTD, batchNormMT) })
	b.FLay(func() common.IHLayer { return nn.NewHLayerRelu() })
	//b.Optimizer(func() common.IOptimizer { return nn.NewOptMomentum(learingRate, optMT) })
	//b.Optimizer(func() common.IOptimizer { return nn.NewOptNormal(learingRate) })
	b.FOpt(func() common.IOptimizer { return nn.NewOptNormal(learingRate) })
	b.Tar(nn.NewTarCE())
	m := b.Build()

	lineChart := sample.NewLineChart("handwritten")
	lineChart.Reg("acc_vali", "acc_test", "loss_train", "loss_vali", "loss_test")
	fmt.Printf("train start\n")
	for e := 0; e < epoch; e++ {
		if e >= lineChartChild {
			m.Trains(trainx, trainy)
		} else {
			sampFreq := int(float64(len(trainx)) / samplingRate)
			m.TrainTimes(trainx, trainy, func(trainTimes, spendMS int) {
				if trainTimes%sampFreq == 0 {
					child := lineChart.Child(e)
					vloss, vacc := m.Tests(valix, valiy)
					tloss, tacc := m.Tests(testx, testy)
					child.Append(vacc, tacc, m.LossLatest(), vloss, tloss)
					fmt.Printf("train at:%d, trainTimes:%d, test info :%s, spend:%d\n", e, trainTimes, child.Format(child.Len()-1), spendMS)
				}
			})
		}
		vloss, vacc := m.Tests(valix, valiy)
		tloss, tacc := m.Tests(testx, testy)
		lineChart.Append(vacc, tacc, m.LossPopMean(), vloss, tloss)
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	fmt.Printf("train end\n")
	lineChart.Draw()
}

func main() {
	go func() {
		fmt.Println(http.ListenAndServe(":6060", nil))
	}()
	handwritten()
}
