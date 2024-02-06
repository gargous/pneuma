package rnn

import "pneuma/nn"

type Model struct {
	*nn.Model
}

func NewModel() *Model {
	ret := &Model{
		Model: nn.NewModel(),
	}
	return ret
}
