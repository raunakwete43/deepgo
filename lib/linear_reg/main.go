package linearreg

import (
	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type WF map[string]*matrix.Matrix

type LinearReg struct {
	W, B *matrix.Matrix
}

func Forward_Linear_Regression(X, y *matrix.Matrix, weights WF) (float64, WF) {
	if X.Rows != y.Rows {
		panic("Batch size not equal")
	}

	if X.Cols != weights["W"].Rows {
		panic("Matmul will not work")
	}

	if weights["B"].Rows != 1 && weights["B"].Cols != 1 {
		panic("B is not a 1x1 matrix")
	}

	N := X.Dot(weights["W"])

	P := N.Add(weights["B"])

	loss := y.Subtract(P).Apply(func(f float64) float64 { return f * f }).Mean()

	forward_info := WF{
		"X": X,
		"N": N,
		"P": P,
		"y": y,
	}

	return loss, forward_info
}

func Loss_Gradients(forward_info, weights WF) WF {
	// batch_size := forward_info["X"].Rows

	dLdP := (forward_info["y"].Subtract(forward_info["P"])).SingleMul(-2)

	dPdN := matrix.OnesLike(forward_info["N"])

	dPdB := matrix.OnesLike(weights["B"])

	dLdN := dLdP.Multiply(dPdN)

	dNdW := forward_info["X"].Transpose()

	dLdW := dNdW.Dot(dLdN)

	dLdB := dLdP.SingleMul(dPdB.Data[0][0]).AxisSum(0)

	loss_gradients := WF{
		"W": dLdW,
		"B": dLdB,
	}

	return loss_gradients
}

func Predict(X *matrix.Matrix, weights WF) *matrix.Matrix {
	N := X.Dot(weights["W"])

	return N.Add(weights["B"])
}
