package sample

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
	"gonum.org/v1/gonum/mat"
)

type NNSample struct {
	X *mat.VecDense
	Y *mat.VecDense
}

func RandSample(trainSamp, testSamp []NNSample) (trainSampi, testSampi, valiSampi []NNSample) {
	trainSampi = make([]NNSample, len(trainSamp))
	testSampi = make([]NNSample, len(testSamp))
	valiSampi = make([]NNSample, len(testSamp))
	for i, tindex := range rand.Perm(len(trainSamp)) {
		trainSampi[i] = trainSamp[tindex]
	}
	for i, tindex := range rand.Perm(len(testSamp)) {
		testSampi[i] = testSamp[tindex]
	}
	for i, tindex := range rand.Perm(len(trainSamp)) {
		if i >= len(testSamp) {
			break
		}
		valiSampi[i] = trainSamp[tindex]
	}
	return
}

func StackSample(trainSamp []NNSample, batch int) (trainx, trainy []*mat.Dense) {
	for i := batch; i < len(trainSamp); i += batch {
		xrow := trainSamp[i].X.Len()
		yrow := trainSamp[i].Y.Len()
		x := mat.NewDense(xrow, batch, nil)
		y := mat.NewDense(yrow, batch, nil)
		cIdx := 0
		for j := i - batch; j < i; j++ {
			sx := trainSamp[j].X.RawVector().Data
			sy := trainSamp[j].Y.RawVector().Data
			x.SetCol(cIdx, sx)
			y.SetCol(cIdx, sy)
			cIdx++
		}
		trainx = append(trainx, x)
		trainy = append(trainy, y)
	}
	return
}

type LineChart struct {
	title    string
	key      []string
	datas    map[string][]float64
	children map[int]*LineChart
}

func NewLineChart(title string) *LineChart {
	return &LineChart{
		title:    title,
		datas:    make(map[string][]float64),
		children: make(map[int]*LineChart),
	}
}

func (l *LineChart) Reg(key ...string) {
	l.key = key
}

func (l *LineChart) Append(val ...float64) {
	for i := 0; i < len(l.key); i++ {
		key := l.key[i]
		l.datas[key] = append(l.datas[key], val[i])
	}
}

func (l *LineChart) Len() int {
	return len(l.datas[l.key[0]])
}

func (l *LineChart) Format(index int) string {
	ret := make([]string, len(l.key))
	for i := 0; i < len(l.key); i++ {
		key := l.key[i]
		val := l.datas[key][index]
		ret[i] = fmt.Sprintf("%s:%.4f", key, val)
	}
	return strings.Join(ret, ",")
}

func (l *LineChart) Child(index int) *LineChart {
	child := l.children[index]
	if child == nil {
		child = NewLineChart(fmt.Sprintf("%s_%d", l.title, index))
		child.Reg(l.key...)
	}
	l.children[index] = child
	return child
}

func (l *LineChart) draw() {
	// create a new line instance
	line := charts.NewLine()
	// set some global options like Title/Legend/ToolTip or anything else
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{
			Theme: types.ThemeInfographic,
			Width: "1200px",
		}),
		charts.WithTitleOpts(opts.Title{
			Title:    l.title,
			Subtitle: l.title,
		}),
	)
	var axis []string
	for name, data := range l.datas {
		items := make([]opts.LineData, len(data))
		for i := 0; i < len(data); i++ {
			items[i] = opts.LineData{Value: data[i]}
		}
		if len(axis) <= 0 {
			axis = make([]string, len(data))
			for i := 0; i < len(data); i++ {
				if i%(len(data)/5) == 0 {
					axis[i] = strconv.Itoa(i)
				} else {
					axis[i] = ""
				}
			}
			line.SetXAxis(axis)
		}
		line.
			AddSeries(name, items).
			SetSeriesOptions(charts.WithLineChartOpts(opts.LineChart{}))
	}
	line.SetSeriesOptions(charts.WithLineChartOpts(opts.LineChart{Smooth: true}))
	f, _ := os.Create(l.title + ".html")
	_ = line.Render(f)
}

func (l *LineChart) Draw() {
	l.draw()
	for _, c := range l.children {
		c.Draw()
	}
}
