package nn

import (
	"math"

	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type Sigmoid struct {
	BaseOperation
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{
		BaseOperation: *NewBaseOperation(),
	}
}

func (self *Sigmoid) _output() *matrix.Matrix {
	return self.input_.Apply(func(f float64) float64 {
		return 1 / (1 + math.Exp(-f))
	})
}

func (self *Sigmoid) _input_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	sigmoid_backward := self.output.Multiply(self.output.Apply(func(f float64) float64 {
		return 1 - f
	}))

	input_grad := sigmoid_backward.Multiply(output_grad)

	return input_grad
}

func (self *Sigmoid) Forward(input_ *matrix.Matrix) *matrix.Matrix {
	self.input_ = input_

	self.output = self._output()

	return self.output
}

// func (self *Sigmoid) Backward(output_grad *matrix.Matrix) *matrix.Matrix {
// 	assert_same_shape(self.output, output_grad)
//
// 	self.input_grad = self._input_grad(output_grad)
//
// 	assert_same_shape(self.input_, self.input_grad)
//
// 	return self.input_grad
// }
//

func test() {
	sm := &Sigmoid{}
	sm.__init__()
	sm.Bac
}
