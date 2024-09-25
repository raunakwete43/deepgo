package nn

import "github.com/raunakwete43/deepgo.git/lib/matrix"

type ParamOperation struct {
	BaseOperation
	param      *matrix.Matrix
	param_grad *matrix.Matrix
}

func (self *ParamOperation) __init__(param *matrix.Matrix) *ParamOperation {
	self.param = param
	return self
}

func (self *ParamOperation) Backward(output_grad *matrix.Matrix) *matrix.Matrix {
	assert_same_shape(self.output, output_grad)

	self.input_grad = self._input_grad(output_grad)
	self.param_grad = self._param_grad(output_grad)

	assert_same_shape(self.input_, self.input_grad)
	assert_same_shape(self.param, self.param_grad)

	return self.input_grad
}

func (self *ParamOperation) _param_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	panic("Not implemented")
}
