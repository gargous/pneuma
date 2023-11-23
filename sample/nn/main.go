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

func main() {
	batch := 100
	trainSamp := make([]nnSample, 100000)
	testSamp := make([]nnSample, 2000)
	lables := []*mat.VecDense{mat.NewVecDense(2, []float64{1, 0}), mat.NewVecDense(2, []float64{0, 1})}
	makeRGBASample(trainSamp, testSamp, lables)
	//makeBTZeroSample(trainSamp, testSamp, lables)
	//makePrimeSample(trainSamp, testSamp, lables)
	trainSamp, testx, testy, valix, valiy := randSample(trainSamp, testSamp)
	var trainx []*mat.Dense
	var trainy []*mat.Dense
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
	builder := nn.NewModelBuilder()
	//builder.Size(trainSamp[0].x.Len(), 10, 10, 2)
	builder.Size(trainSamp[0].x.Len(), 16, 10, 2)
	//builder.Optimizer(func() nn.IOptimizer { return nn.NewOptNormal(0.01) })
	builder.Optimizer(func() nn.IOptimizer { return nn.NewOptMomentum(0.01, 0.1) })
	builder.LayerAt(func(r, c int) nn.IHLayer { return nn.NewHLayerBatchNorm(r, 0.0001, 0.9) })
	//builder.Layer(func() nn.IHLayer { return nn.NewHLayerRelu() })
	builder.Layer(func() nn.IHLayer { return nn.NewHLayerSigmoid() })
	builder.Target(nn.NewTarCE(), &nn.LossParam{
		Threshold: 0.01,
		MinLoss:   0.01,
		MinTimes:  1000,
	})

	m := builder.Build()
	items := make(map[string][]float64)
	itemses := make([]map[string][]float64, 16)
	itemsesShow := map[int]bool{0: true, 1: true}
	samplingRate := 5.0

	for e := 0; e < len(itemses); e++ {
		itemses[e] = make(map[string][]float64)
		testRate := samplingRate
		if !itemsesShow[e] {
			testRate = 0
		}
		retInfo, testInfos := m.TrainEpoch(trainx, trainy, testx, testy, valix, valiy, testRate)
		for _, testInfo := range testInfos {
			itemses[e]["acc_vali"] = append(itemses[e]["acc_vali"], testInfo.ValiAcc)
			itemses[e]["acc_test"] = append(itemses[e]["acc_test"], testInfo.TestAcc)
			itemses[e]["loss"] = append(itemses[e]["loss"], testInfo.Loss)
		}
		fmt.Printf("train at:%d, %+v\n", e, retInfo)
		itemses[e]["acc_vali"] = append(itemses[e]["acc_vali"], retInfo.ValiAcc)
		itemses[e]["acc_test"] = append(itemses[e]["acc_test"], retInfo.TestAcc)
		itemses[e]["loss"] = append(itemses[e]["loss"], retInfo.Loss)
		m.LossDrop()
	}
	sample.LineChart("nn", items)
	for k := range itemsesShow {
		sample.LineChart(fmt.Sprintf("nn%d", k), itemses[k])
	}
}
