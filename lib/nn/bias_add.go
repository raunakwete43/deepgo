package nn

import "github.com/raunakwete43/deepgo.git/lib/matrix"

type BiasAdd struct {
	ParamOperation
}

func NewBiasAdd(b *matrix.Matrix) *BiasAdd {
	if b.Rows != 1 {
		panic("Shape of B must be (1, _)")
	}

	return &BiasAdd{
		ParamOperation: *NewParamOperation(b),
	}
}

func (self *BiasAdd) _output() *matrix.Matrix {
	return self.input_.Add(self.param)
}

func (self *BiasAdd) _input_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	return matrix.OnesLike(self.input_).Multiply(output_grad)
}

func (self *BiasAdd) _param_grad(output_grad *matrix.Matrix) *matrix.Matrix {
	param_grad := matrix.OnesLike(self.param).Multiply(output_grad)

	return param_grad.AxisSum(0).Reshape(1, param_grad.Cols)
}

func (self *BiasAdd) Forward(input_ *matrix.Matrix) *matrix.Matrix {
	self.input_ = input_

	self.output = self._output()

	return self.output
}

func (self *BiasAdd) Backward(output_grad *matrix.Matrix) *matrix.Matrix {
	assert_same_shape(self.output, output_grad)

	self.input_grad = self._input_grad(output_grad)
	self.param_grad = self._param_grad(output_grad)

	assert_same_shape(self.input_, self.input_grad)
	assert_same_shape(self.param, self.param_grad)

	return self.input_grad
}
