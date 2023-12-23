package cnn

import (
	"image"
	"image/color"
	"pneuma/common"
)

func ImgToVecData(img image.Image, channel int) ([]float64, []int) {
	b := img.Bounds().Size()
	c := b.X
	r := b.Y
	m := channel
	ret := make([]float64, c*r*m)
	onecSize := []int{r, c}
	onecSizeLen := common.IntsProd(onecSize)
	common.RecuRange(onecSize, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		idx := common.PosIdx(pos, onecSize)
		colr := img.At(j, i)
		cr, cg, cb, ca := colr.RGBA()
		cs := []float64{float64(cr) / 255.0, float64(cg) / 255.0, float64(cb) / 255.0, float64(ca) / 255.0}
		for k := 0; k < m; k++ {
			ret[common.IdxExpend(k, idx, onecSizeLen)] = cs[k]
		}
	})
	return ret, append(onecSize, m)
}

func VecDataToImage(data []float64, size []int) image.Image {
	r, c, m := size[0], size[1], size[2]
	ret := image.NewRGBA(image.Rect(0, 0, c, r))
	onecSize := size[:2]
	onecSizeLen := common.IntsProd(onecSize)
	common.RecuRange(onecSize, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		idx := common.PosIdx(pos, onecSize)
		cs := make([]uint8, m)
		for k := 0; k < m; k++ {
			cs[k] = uint8(data[common.IdxExpend(k, idx, onecSizeLen)] * 255.0)
		}
		switch m {
		case 1:
			ret.Set(j, i, color.Gray{cs[0]})
		case 3:
			ret.Set(j, i, color.RGBA{cs[0], cs[1], cs[2], 255})
		case 4:
			ret.Set(j, i, color.RGBA{cs[0], cs[1], cs[2], cs[3]})
		}
	})
	return ret
}
