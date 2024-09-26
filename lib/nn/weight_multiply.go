package nn

import "github.com/raunakwete43/deepgo.git/lib/matrix"

type WeightMultiply struct {
	ParamOperation
}

func NewWeightMultiply(w *matrix.Matrix) *WeightMultiply {
	return &WeightMultiply{
		ParamOperation: *NewParamOperation(w),
	}
}

func (self *WeightMultiply) _output() *matrix.Matrix {
	return self.input_.Dot(self.param)
}

func (self *WeightMultiply) _input_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	return output_grad.Dot(self.param.Transpose())
}

func (self *WeightMultiply) _param_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	return self.input_.Transpose().Dot(output_grad)
}

func (self *WeightMultiply) Forward(input_ *matrix.Matrix) *matrix.Matrix {
	self.input_ = input_

	self.output = self._output()

	return self.output
}

func (self *WeightMultiply) Backward(output_grad *matrix.Matrix) *matrix.Matrix {
	assert_same_shape(self.output, output_grad)

	self.input_grad = self._input_grad(output_grad)
	self.param_grad = self._param_grad(output_grad)

	assert_same_shape(self.input_, self.input_grad)
	assert_same_shape(self.param, self.param_grad)

	return self.input_grad
}
