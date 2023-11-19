package main

import (
	"fmt"
	"pneuma/nn"
	"pneuma/sample"

	"gonum.org/v1/gonum/mat"
)

func randSample(trainSamp, testSamp []nnSample) (trainSampi []nnSample, testx, testy, valix, valiy *mat.Dense) {
	trainSampi = make([]nnSample, len(trainSamp))
	for i, tindex := range sample.RandPerm(len(trainSamp)) {
		trainSampi[i] = trainSamp[tindex]
	}
	testSampi := make([]nnSample, len(testSamp))
	for i, tindex := range sample.RandPerm(len(testSamp)) {
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
	batch := 50
	trainSamp := make([]nnSample, 100000)
	testSamp := make([]nnSample, 2000)
	lables := []*mat.VecDense{mat.NewVecDense(2, []float64{1, 0}), mat.NewVecDense(2, []float64{0, 1})}
	//makeBTZeroSample(trainSamp, testSamp, lables)
	makeRGBASample(trainSamp, testSamp, lables)
	//makePrimeSample(trainSamp, testSamp, lables)
	//makeKRKSample(trainSamp, testSamp, lables)
	trainSamp, testx, testy, valix, valiy := randSample(trainSamp, testSamp)

	m := nn.NewStdModel(
		[]int{trainSamp[0].x.Len(), 10, 10, 2},
		func(r, c int) (nn.IOptimizer, []nn.IHLayer) {
			return nn.NewOptMomentum(0.01, 0.1), []nn.IHLayer{
				nn.NewHLayerBatchNorm(r, 0.0001, 0.9),
				nn.NewHLayerSigmoid(),
			}
		})
	m.SetTarget(nn.NewTarCE(), &nn.LossParam{
		Threshold: 0.01,
		MinLoss:   0.01,
		MinTimes:  1000,
	})

	trainTimes := 0
	items := make(map[string][]float64)
	for e := 0; e < 50; e++ {
		var trainTimesE int
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
			m.Train(x, y)
			if m.IsDone() {
				break
			}
			trainTimes++
			trainTimesE++
		}
		valiAcc := m.Test(valix, valiy)
		testAcc := m.Test(testx, testy)
		loss := m.LossMean()
		fmt.Printf("train at:%d, valiAcc:%.4f,testAcc:%.4f,loss:%.4f", e, valiAcc, testAcc, loss)
		items["acc_vali"] = append(items["acc_vali"], valiAcc)
		items["acc_test"] = append(items["acc_test"], testAcc)
		items["loss"] = append(items["loss"], loss)
		m.LossDrop()
	}
	sample.LineChart("nn", items)
}
