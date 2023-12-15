package main

import (
	"fmt"
	"pneuma/common"
	"pneuma/dnn"
	"pneuma/nn"
	"pneuma/sample"

	"gonum.org/v1/gonum/mat"
)

func main() {
	epoch := 16
	batch := 100
	samplingRate := 100.0
	lineChartChild := 2
	trainSamp := make([]sample.NNSample, 10000)
	testSamp := make([]sample.NNSample, 200)
	lables := []*mat.VecDense{mat.NewVecDense(2, []float64{1, 0}), mat.NewVecDense(2, []float64{0, 1})}
	makeKRKSample(trainSamp, testSamp, lables)
	//makeBT5Sample(trainSamp, testSamp, lables)
	//makeBTZeroSample(trainSamp, testSamp, lables)
	//makePrimeSample(trainSamp, testSamp, lables, true)
	trainSamp, testx, testy, valix, valiy := sample.RandSample(trainSamp, testSamp)
	trainx, trainy := sample.StackSample(trainSamp, batch)

	builder := dnn.NewModelBuilder()
	//builder.Size(trainSamp[0].x.Len(), 10, 10, 2)
	builder.Size(trainSamp[0].X.Len(), 16, 10, 2)
	//builder.Optimizer(func() nn.IOptimizer { return nn.NewOptNormal(0.01) })
	builder.Optimizer(func() common.IOptimizer { return nn.NewOptMomentum(0.01, 0.1) })
	builder.Layer(func() common.IHLayer { return nn.NewHLayerBatchNorm(0.0001, 0.9) })
	//builder.Layer(func() nn.IHLayer { return nn.NewHLayerRelu() })
	builder.Layer(func() common.IHLayer { return nn.NewHLayerSigmoid() })
	builder.Target(func() common.ITarget { return nn.NewTarCE() })

	m := builder.Build()
	nn.NewIniSAE(m).Init(trainx)

	lineChart := sample.NewLineChart("krk")
	lineChart.Reg("acc_vali", "acc_test", "loss_train")
	for e := 0; e < epoch; e++ {
		if e >= lineChartChild {
			m.TrainEpoch(trainx, trainy)
		} else {
			sampFreq := int(float64(len(trainx)) / samplingRate)
			m.TrainEpochTimes(trainx, trainy, func(trainTimes int) {
				if trainTimes%sampFreq == 0 {
					vpred := m.Predict(valix)
					tpred := m.Predict(testx)
					lineChart.Child(e).Append(m.Acc(vpred, valiy), m.Acc(tpred, testy), m.LossLatest())
				}
			})
		}
		vpred := m.Predict(valix)
		tpred := m.Predict(testx)
		lineChart.Append(m.Acc(vpred, valiy), m.Acc(tpred, testy), m.LossPopMean())
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	lineChart.Draw()
}
