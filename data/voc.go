package data

import (
	"bufio"
	"encoding/xml"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"
	"pneuma/common"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type VOCSet struct {
	labelNames []string
	labelMat   *mat.Dense
	Trains     *VOCDatas
	Valids     *VOCDatas
	Tests      *VOCDatas
	annotPath  string
	RootPath   string
	size       []int
}

func NewVOCSet(size []int, path string) *VOCSet {
	return &VOCSet{
		RootPath:  path,
		size:      size,
		annotPath: filepath.Join(path, "Annotations"),
		Trains:    NewVOCDatas(path, "train.txt", size),
		Tests:     NewVOCDatas(path, "test.txt", size),
		Valids:    NewVOCDatas(path, "train.txt", size),
	}
}

func ReadLine(fileName string, cb func(string)) error {
	r, err := os.Open(fileName)
	if err != nil {
		return errors.Join(errors.New("open file"), err)
	}
	defer r.Close()
	buff := bufio.NewReader(r)
	for {
		line, err := buff.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			} else {
				return errors.Join(errors.New("read string"), err)
			}
		}
		cb(line)
	}
}

func (v *VOCSet) ReadAnnot() error {
	annotDirs, err := os.ReadDir(v.annotPath)
	if err != nil {
		return errors.Join(errors.New("read dir"), err)
	}
	datas := make(map[string]*VOCData, len(annotDirs))
	for _, annotDir := range annotDirs {
		fname := annotDir.Name()
		fpath := filepath.Join(v.annotPath, fname)
		annotData, err := os.ReadFile(fpath)
		if err != nil {
			return errors.Join(fmt.Errorf("read file at %s", fpath), err)
		}
		excIdx := strings.Index(fname, ".")
		if excIdx < 0 {
			return fmt.Errorf("file name %s has no exc", fname)
		}
		if string(fname[excIdx+1:]) != "xml" {
			continue
		}
		vocData := &VOCData{}
		err = vocData.parseAnnot(annotData, fname[:excIdx], v.size)
		if err != nil {
			return errors.Join(fmt.Errorf("parse annot at %s", fname), err)
		}
		datas[vocData.Idx] = vocData
		for _, obj := range vocData.Objects {
			obj.Lable = -1
			for labIdx, labName := range v.labelNames {
				if strings.EqualFold(labName, obj.Name) {
					obj.Lable = labIdx
					break
				}
			}
			if obj.Lable < 0 {
				obj.Lable = len(v.labelNames)
				v.labelNames = append(v.labelNames, obj.Name)
			}
		}
	}
	err = v.Trains.SelectAnnot(datas)
	if err != nil {
		return errors.Join(errors.New("select train annot"), err)
	}
	err = v.Tests.SelectAnnot(datas)
	if err != nil {
		return errors.Join(errors.New("select tests annot"), err)
	}
	err = v.Valids.SelectAnnot(datas)
	if err != nil {
		return errors.Join(errors.New("select valid annot"), err)
	}
	v.labelMat = mat.NewDense(len(v.labelNames), len(v.labelNames), nil)
	for i, _ := range v.labelNames {
		v.labelMat.Set(i, i, 1)
	}
	return nil
}

func (v *VOCSet) Label(idx int) []float64 {
	return v.labelMat.RawRowView(idx)
}

type VOCDatas struct {
	size    []int
	datas   []*VOCData
	loadAt  int
	IdxPath string
	imgPath string
}

func NewVOCDatas(path, idxFileName string, size []int) *VOCDatas {
	return &VOCDatas{
		size:    size,
		imgPath: filepath.Join(path, "JPEGImages"),
		IdxPath: filepath.Join(path, "ImageSets", "Main", idxFileName),
	}
}

func (v *VOCDatas) SelectAnnot(datas map[string]*VOCData) error {
	v.datas = make([]*VOCData, len(datas))
	idx := 0
	err := ReadLine(v.IdxPath, func(s string) {
		v.datas[idx] = datas[strings.TrimSpace(s)]
		idx++
	})
	if err != nil {
		return errors.Join(errors.New("ReadLine"), err)
	}
	v.datas = v.datas[:idx:idx]
	return nil
}
func (v *VOCDatas) Len() int {
	return len(v.datas)
}

func (v *VOCDatas) Reindex(idx []int) {
	oldData := v.datas
	v.datas = make([]*VOCData, len(idx))
	for i := 0; i < len(idx); i++ {
		v.datas[i] = oldData[idx[i]]
	}
}

func (v *VOCDatas) DataToBnd(data *VOCData) []float64 {
	size := (len(v.size) - 1) * 2
	objs := data.Objects
	ret := make([]float64, len(objs)*size)
	for i := 0; i < len(objs); i++ {
		idx := i * size
		copy(ret[idx:idx+size], objs[i].Bound)
	}
	return ret
}

func (v *VOCDatas) load(cnt int) ([]*VOCData, error) {
	end := common.IntsMin(v.loadAt+cnt, len(v.datas))
	ret := make([]*VOCData, end-v.loadAt)
	for i := 0; i < len(ret); i++ {
		data := &VOCData{}
		data.copyAnnot(v.datas[v.loadAt+i])
		imgPath := filepath.Join(v.imgPath, data.ImgName)
		imgReader, err := os.Open(imgPath)
		if err != nil {
			return nil, err
		}
		err = data.decodeImg(imgReader, v.size)
		if err != nil {
			return nil, err
		}
		imgReader.Close()
		ret[i] = data
	}
	v.loadAt = end
	return ret, nil
}

func (v *VOCDatas) Pop(cnt int) ([]*VOCData, error) {
	return v.load(cnt)
}

func (v *VOCDatas) Load(cnt int) ([]*VOCData, error) {
	return v.load(cnt)
}

func (v *VOCDatas) Annots() []*VOCData {
	return v.datas
}

func (v *VOCDatas) ResetLoad() {
	v.loadAt = 0
}

type VOCObject struct {
	Bound []float64
	Name  string
	Lable int
}

type VOCData struct {
	Img     []float64
	Objects []VOCObject
	Idx     string
	ImgName string
}

type vocXMLSize struct {
	Width  int `xml:"width"`
	Height int `xml:"height"`
	Depth  int `xml:"depth"`
}

type vocXMLBox struct {
	XMin int `xml:"xmin"`
	YMin int `xml:"ymin"`
	XMax int `xml:"xmax"`
	YMax int `xml:"ymax"`
}

type vocXMLObject struct {
	Name      string    `xml:"name"`
	Pose      string    `xml:"pose"`
	Truncated int       `xml:"truncated"`
	Bound     vocXMLBox `xml:"bndbox"`
}

type vocXMLAnnotation struct {
	FileName string         `xml:"filename"`
	Size     vocXMLSize     `xml:"size"`
	Object   []vocXMLObject `xml:"object"`
}

func (v *VOCData) copyAnnot(src *VOCData) {
	v.Objects = src.Objects
	v.Idx = src.Idx
	v.ImgName = src.ImgName
}

func (v *VOCData) parseAnnot(annotData []byte, idxName string, size []int) error {
	a := vocXMLAnnotation{}
	err := xml.Unmarshal(annotData, &a)
	if err != nil {
		return errors.Join(errors.New("xml unmarshal"), err)
	}
	v.Objects = make([]VOCObject, len(a.Object))
	v.ImgName = a.FileName
	v.Idx = idxName
	orgSize := []float64{float64(a.Size.Height), float64(a.Size.Width)}
	whlCtr := make([]float64, 2)
	bndCtr := make([]float64, 2)
	bndRadius := make([]float64, 2)
	scales := common.IntsToF64s(size[0:2])
	floats.Div(scales, orgSize)
	floats.Scale(0.5, orgSize)
	for i, aObj := range a.Object {
		bndi := aObj.Bound
		bndf := []float64{float64(bndi.YMin), float64(bndi.XMin), float64(bndi.YMax), float64(bndi.XMax)}
		floats.AddTo(bndCtr, bndf[2:4], bndf[0:2])
		floats.Scale(0.5, bndCtr)
		floats.SubTo(bndRadius, bndf[2:4], bndf[0:2])
		floats.Scale(0.5, bndRadius)
		floats.Sub(bndCtr, whlCtr)
		floats.Mul(bndRadius, scales)
		floats.Mul(bndCtr, scales)
		floats.Add(bndCtr, whlCtr)
		floats.SubTo(bndf[0:2], bndCtr, bndRadius)
		floats.AddTo(bndf[2:4], bndCtr, bndRadius)
		v.Objects[i] = VOCObject{
			Bound: bndf,
			Name:  aObj.Name,
		}
	}
	return nil
}

func (v *VOCData) decodeImg(r io.Reader, size []int) error {
	img, err := jpeg.Decode(r)
	if err != nil {
		return err
	}

	v.Img = ImgToVecData(img, size)
	return nil
}

func DrawBnd(img image.Image, bound []float64, bndWidth int) {
	rgba := img.(draw.Image)
	red := image.NewUniform(color.RGBA{255, 0, 0, 255})
	xmin, ymin := int(bound[1]), int(bound[0])
	xmax, ymax := int(bound[3]), int(bound[2])
	top := image.Rect(xmin, ymin, xmax, ymin+bndWidth)
	bot := image.Rect(xmin, ymax-bndWidth, xmax, ymax)
	lef := image.Rect(xmin, ymin, xmin+bndWidth, ymax)
	rig := image.Rect(xmax-bndWidth, ymin, xmax, ymax)
	draw.Draw(rgba, top, red, image.Point{}, draw.Src)
	draw.Draw(rgba, bot, red, image.Point{}, draw.Src)
	draw.Draw(rgba, lef, red, image.Point{}, draw.Src)
	draw.Draw(rgba, rig, red, image.Point{}, draw.Src)
}

func (v *VOCData) BndImg(w io.Writer, size []int) error {
	img := VecDataToImage(v.Img, size)
	for _, o := range v.Objects {
		DrawBnd(img, o.Bound, 3)
	}
	return jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
}
