package sample

import (
	"math/rand"
	"os"
	"strconv"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
)

func RandPerm(n int) []int {
	v := make([]int, n)
	for i := 0; i < n; i++ {
		v[i] = i
	}
	for i := 0; i < n; i++ {
		index := rand.Intn(n)
		oldV := v[index]
		v[index] = v[i]
		v[i] = oldV
	}
	return v
}

func LineChart(title string, datas map[string][]float64) {
	// create a new line instance
	line := charts.NewLine()
	// set some global options like Title/Legend/ToolTip or anything else
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{
			Theme: types.ThemeInfographic,
			Width: "1200px",
		}),
		charts.WithTitleOpts(opts.Title{
			Title:    title,
			Subtitle: title,
		}),
	)
	var axis []string
	for name, data := range datas {
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
	f, _ := os.Create(title + ".html")
	_ = line.Render(f)
}
