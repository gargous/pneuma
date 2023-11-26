package main

import (
	"fmt"
	"math/rand"
	"pneuma/nn"
	"pneuma/sample"

	"gonum.org/v1/gonum/mat"
)

func randSample(trainSamp, testSamp []nnSample) (trainSampi []nnSample, testx, testy, valix, valiy *mat.Dense) {
	trainSampi = make([]nnSample, len(trainSamp))
	for i, tindex := range rand.Perm(len(trainSamp)) {
		trainSampi[i] = trainSamp[tindex]
	}
	testSampi := make([]nnSample, len(testSamp))
	for i, tindex := range rand.Perm(len(testSamp)) {
		testSampi[i] = testSamp[tindex]
	}
	testSamp = testSampi
	testx = mat.NewDense(trainSamp[0].x.Len(), len(testSamp), nil)
	testy = mat.NewDense(trainSamp[0].y.Len(), len(testSamp), nil)
	valix = mat.NewDense(trainSamp[0].x.Len(), len(testSamp), nil)
	valiy = mat.NewDense(trainSamp[0].y.Len(), len(testSamp), nil)

	for j := 0; j < testx.RawMatrix().Cols; j++ {
		testx.SetCol(j, testSamp[j].x.RawVector().Data)
		testy.SetCol(j, testSamp[j].y.RawVector().Data)
	}
	for j := 0; j < valix.RawMatrix().Cols; j++ {
		valix.SetCol(j, trainSampi[j].x.RawVector().Data)
		valiy.SetCol(j, trainSampi[j].y.RawVector().Data)
	}
	return
}

func stackSample(trainSamp []nnSample, batch int) (trainx, trainy []*mat.Dense) {
	for i := 0; i < len(trainSamp); i += batch {
		xrow := trainSamp[i].x.Len()
		yrow := trainSamp[i].y.Len()
		x := mat.NewDense(xrow, batch, nil)
		y := mat.NewDense(yrow, batch, nil)
		for j := 0; j < batch; j++ {
			sx := trainSamp[i+j].x.RawVector().Data
			sy := trainSamp[i+j].y.RawVector().Data
			x.SetCol(j, sx)
			y.SetCol(j, sy)
		}
		trainx = append(trainx, x)
		trainy = append(trainy, y)
	}
	return
}

func main() {
	epoch := 16
	batch := 100
	samplingRate := 100.0
	lineChartChild := 2
	trainSamp := make([]nnSample, 100000)
	testSamp := make([]nnSample, 2000)
	lables := []*mat.VecDense{mat.NewVecDense(2, []float64{1, 0}), mat.NewVecDense(2, []float64{0, 1})}
	makeRGBASample(trainSamp, testSamp, lables)
	//makeBTZeroSample(trainSamp, testSamp, lables)
	//makePrimeSample(trainSamp, testSamp, lables, true)
	trainSamp, testx, testy, valix, valiy := randSample(trainSamp, testSamp)
	trainx, trainy := stackSample(trainSamp, batch)

	builder := nn.NewModelBuilder()
	//builder.Size(trainSamp[0].x.Len(), 10, 10, 2)
	builder.Size(trainSamp[0].x.Len(), 16, 10, 2)
	//builder.Optimizer(func() nn.IOptimizer { return nn.NewOptNormal(0.01) })
	builder.Optimizer(func() nn.IOptimizer { return nn.NewOptMomentum(0.01, 0.1) })
	builder.LayerAt(func(r, c int) nn.IHLayer { return nn.NewHLayerBatchNorm(r, 0.0001, 0.9) })
	//builder.Layer(func() nn.IHLayer { return nn.NewHLayerRelu() })
	builder.Layer(func() nn.IHLayer { return nn.NewHLayerSigmoid() })
	builder.Target(func() nn.ITarget { return nn.NewTarCE() })

	m := builder.Build()
	nn.NewIniSAE(m).Init(trainx)

	lineChart := sample.NewLineChart("nn")
	lineChart.Reg("acc_vali", "acc_test", "loss")
	for e := 0; e < epoch; e++ {
		if e >= lineChartChild {
			m.TrainEpoch(trainx, trainy)
		} else {
			sampFreq := int(float64(len(trainx)) / samplingRate)
			m.TrainEpochTimes(trainx, trainy, func(trainTimes int) {
				if trainTimes%sampFreq == 0 {
					lineChart.Child(e).Append(m.Test(valix, valiy), m.Test(testx, testy), m.LossLatest())
				}
			})
		}
		lineChart.Append(m.Test(valix, valiy), m.Test(testx, testy), m.LossPopMean())
		fmt.Printf("train at:%d, %s\n", e, lineChart.Format(lineChart.Len()-1))
	}
	lineChart.Draw()
}
