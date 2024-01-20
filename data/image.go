package data

import (
	"image"
	"image/color"
	"pneuma/common"

	"golang.org/x/image/draw"
)

func ImgToVecData(img image.Image, size []int) []float64 {
	b := img.Bounds().Size()
	r, c, m := size[0], size[1], size[2]
	if b.Y != r || b.X != c {
		dst := image.NewRGBA(image.Rect(0, 0, c, r))
		draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)
		img = dst
	}
	ret := make([]float64, c*r*m)
	onecSize := []int{r, c}
	cmod := color.RGBAModel
	common.RecuRange(onecSize, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		idx := common.PosIdx(pos, onecSize)
		colr := cmod.Convert(img.At(j, i)).(color.RGBA)
		cr, cg, cb, ca := colr.R, colr.G, colr.B, colr.A
		cs := []float64{float64(cr) / 255.0, float64(cg) / 255.0, float64(cb) / 255.0, float64(ca) / 255.0}
		for k := 0; k < m; k++ {
			ret[common.IdxExpend(idx, k, m)] = cs[k]
		}
	})
	return ret
}

func VecDataToImage(data []float64, size []int) image.Image {
	r, c, m := size[0], size[1], size[2]
	ret := image.NewRGBA(image.Rect(0, 0, c, r))
	onecSize := size[:2]
	common.RecuRange(onecSize, nil, func(pos []int) {
		i, j := pos[0], pos[1]
		idx := common.PosIdx(pos, onecSize)
		cs := make([]uint8, m)
		for k := 0; k < m; k++ {
			cs[k] = uint8(data[common.IdxExpend(idx, k, m)] * 255.0)
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
