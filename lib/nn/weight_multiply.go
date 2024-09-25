package nn

import "github.com/raunakwete43/deepgo.git/lib/matrix"

type WeightMultiply struct {
	ParamOperation
}

func (self *WeightMultiply) __init__(w *matrix.Matrix) *WeightMultiply {
	self.ParamOperation.__init__(w)

	return self
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
