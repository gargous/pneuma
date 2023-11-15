package nn

import "gonum.org/v1/gonum/mat"

type Model struct {
	layers []*Layer
	inputs []*mat.Dense
}
