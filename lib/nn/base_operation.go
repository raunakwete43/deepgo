package nn

import "github.com/raunakwete43/deepgo.git/lib/matrix"

type Operation interface {
	Forward(*matrix.Matrix) *matrix.Matrix
	Backward(*matrix.Matrix) *matrix.Matrix
	_output() *matrix.Matrix
	_input_grad(*matrix.Matrix) *matrix.Matrix
	_param_grad(*matrix.Matrix) *matrix.Matrix
}

type BaseOperation struct {
	input_     *matrix.Matrix
	output     *matrix.Matrix
	input_grad *matrix.Matrix
}

func NewBaseOperation() *BaseOperation {
	return &BaseOperation{}
}

func (self *BaseOperation) Forward(input_ *matrix.Matrix) *matrix.Matrix {
	self.input_ = input_

	self.output = self._output()

	return self.output
}

func (self *BaseOperation) Backward(output_grad *matrix.Matrix) *matrix.Matrix {
	assert_same_shape(self.output, output_grad)

	self.input_grad = self._input_grad(output_grad)

	assert_same_shape(self.input_, self.input_grad)

	return self.input_grad
}

func (self *BaseOperation) _output() *matrix.Matrix {
	panic("Not implemented")
}

func (self *BaseOperation) _input_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	panic("Not implemented")
}
