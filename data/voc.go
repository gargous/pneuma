package data

import (
	"encoding/xml"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"
	"pneuma/common"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type VOCSet struct {
	labelNames []string
	labelMat   *mat.Dense
	datas      []*VOCData
	loadAt     int
	annotPath  string
	imgPath    string
}

func NewVOCSet(path string) *VOCSet {
	return &VOCSet{
		annotPath: filepath.Join(path, "Annotations"),
		imgPath:   filepath.Join(path, "JEPGImages"),
	}
}

func (v *VOCSet) ReadAnnot() error {
	annotDirs, err := os.ReadDir(v.annotPath)
	if err != nil {
		return err
	}
	v.datas = make([]*VOCData, len(annotDirs))
	for i, annotDir := range annotDirs {
		fname := filepath.Join(v.annotPath, annotDir.Name())
		annotData, err := os.ReadFile(fname)
		if err != nil {
			return err
		}
		vocData := &VOCData{}
		err = vocData.parseAnnot(annotData)
		if err != nil {
			return err
		}
		v.datas[i] = vocData
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
	v.labelMat = mat.NewDense(len(v.labelNames), len(v.labelNames), nil)
	for i, _ := range v.labelNames {
		v.labelMat.Set(i, i, 1)
	}
	return nil
}

func (v *VOCSet) Label(idx int) []float64 {
	return v.labelMat.RawRowView(idx)
}

func (v *VOCSet) load(cnt int) error {
	end := common.IntsMin(v.loadAt+cnt, len(v.datas))
	for i := v.loadAt; i < end; i++ {
		data := v.datas[i]
		imgPath := filepath.Join(v.imgPath, data.ImgName)
		imgReader, err := os.Open(imgPath)
		if err != nil {
			return err
		}
		err = v.datas[i].decodeImg(imgReader)
		if err != nil {
			return err
		}
	}
	v.loadAt = end
	return nil
}

func (v *VOCSet) Pop(cnt int) ([]*VOCData, error) {
	for {
		if v.loadAt >= len(v.datas) {
			break
		}
		if v.loadAt >= cnt {
			break
		}
		err := v.load(cnt * 2)
		if err != nil {
			return nil, err
		}
	}
	popCnt := common.IntsMin(cnt, len(v.datas))
	data := v.datas[:popCnt]
	v.datas = v.datas[popCnt:]
	v.loadAt -= popCnt
	return data, nil
}

type VOCObject struct {
	Bound common.Rectangle
	Name  string
	Lable int
}

type VOCData struct {
	Size    []int
	Img     []float64
	Objects map[int]VOCObject
	ImgName string
}

type vocSize struct {
	Width  int `xml:"width"`
	Height int `xml:"height"`
	Depth  int `xml:"depth"`
}

type vocBox struct {
	XMin int `xml:"xmin"`
	YMin int `xml:"ymin"`
	XMax int `xml:"xmax"`
	YMax int `xml:"ymax"`
}

type vocObject struct {
	Name      string `xml:"name"`
	Pose      int    `xml:"pose"`
	Depth     int    `xml:"depth"`
	Truncated int    `xml:"truncated"`
	Bound     vocBox `xml:"bndbox"`
}

type vocAnnotation struct {
	FileName string      `xml:"filename"`
	Size     vocSize     `xml:"size"`
	Object   []vocObject `xml:"object"`
}

func (v *VOCData) parseAnnot(annotData []byte) error {
	a := vocAnnotation{}
	err := xml.Unmarshal(annotData, &a)
	if err != nil {
		return err
	}
	v.Size = []int{a.Size.Height, a.Size.Width, a.Size.Depth}
	v.Objects = make(map[int]VOCObject, len(a.Object))
	v.ImgName = a.FileName
	for idx, aObj := range a.Object {
		bound := aObj.Bound
		v.Objects[idx] = VOCObject{
			Bound: common.Rect(bound.XMin, bound.YMin, bound.XMax, bound.YMax),
			Name:  aObj.Name,
		}
	}
	return nil
}

func (v *VOCData) decodeImg(r io.Reader) error {
	img, err := jpeg.Decode(r)
	if err != nil {
		return err
	}
	v.Img, _ = ImgToVecData(img, v.Size[len(v.Size)-1])
	return nil
}
