package linearreg

import (
	"math"
	"math/rand"

	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type WF map[string]*matrix.Matrix

type LinearReg struct {
	W, B *matrix.Matrix
}

type Batch struct {
	X, Y *matrix.Matrix
	Size int
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

func loss_gradients(forward_info, weights WF) WF {
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

func (l *LinearReg) Predict(X *matrix.Matrix) *matrix.Matrix {
	N := X.Dot(l.W)

	return N.Add(l.B)
}

func Init_Linear(d *Dataset, g *rand.Rand) *LinearReg {
	return &LinearReg{
		W: matrix.Random(d.N_Feat, 1, g),
		B: matrix.Random(1, 1, g),
	}
}

// func (l *LinearReg) Batch_Fit(d *Dataset, num_epochs int, lr float64, batch_size int) *LinearReg {
// 	batches := d.X.Rows / batch_size
// 	num_epochs /= batches
//
// 	for i := range batches {
// 		X, y := d.X.Data[i*batch_size:(i+1)*batch_size-1], d.Y.Data[i*batch_size:(i+1)*batch_size-1]
// 		l = l.Fit(matrix.InitMatrix(X), matrix.InitMatrix(y), num_epochs, lr)
// 	}
// 	return l
// }

func (l *LinearReg) Fit(d *Dataset, num_epochs int, lr float64) *LinearReg {
	weights := WF{
		"W": l.W,
		"B": l.B,
	}
	for range num_epochs {
		_, forward_info := Forward_Linear_Regression(d.X, d.Y, weights)

		loss_gradients := loss_gradients(forward_info, weights)

		for key := range weights {
			weights[key] = weights[key].Subtract(loss_gradients[key].SingleMul(lr))
		}
	}

	return &LinearReg{
		W: weights["W"],
		B: weights["B"],
	}
}

func (l *LinearReg) RMSE(X, y *matrix.Matrix) float64 {
	if X.Rows != y.Rows {
		panic("X and y must have same number of rows")
	}

	predicted := l.Predict(X)
	diff := predicted.Subtract(y)
	sumSqauredError := diff.Apply(func(f float64) float64 { return f * f }).Mean()

	return float64(math.Sqrt(float64(sumSqauredError)))
}

// func init_weights(input_size, hidden_size int, g *rand.Rand) WF {
// 	weights := WF{
// 		"W1": matrix.Random(input_size, hidden_size, g),
// 		"B1": matrix.Random(1, hidden_size, g),
// 		"W2": matrix.Random(hidden_size, 1, g),
// 		"B2": matrix.Random(1, 1, g),
// 	}
//
// 	return weights
// }
//
// func Forward_Loss(X, y *matrix.Matrix, weights WF) (WF, float64) {
// 	M1 := X.Dot(weights["W1"])
//
// 	N1 := M1.Add(weights["B1"])
//
// 	O1 := N1.Apply(func(f float64) float64 { return sigmoid(f) })
//
// 	M2 := O1.Dot(weights["W2"])
//
// 	P := M2.Add(weights["B2"])
//
// 	loss := y.Subtract(P).Apply(func(f float64) float64 { return f * f }).Mean()
//
// 	forward_info := WF{
// 		"X":  X,
// 		"M1": M1,
// 		"N1": N1,
// 		"O1": O1,
// 		"M2": M2,
// 		"P":  P,
// 		"y":  y,
// 	}
//
// 	return forward_info, loss
// }
//
// func Loss_Gradients(forward_info, weights WF) WF {
// 	dLdP := forward_info["y"].Subtract(forward_info["P"]).Apply(func(f float64) float64 { return -f })
//
// 	dPdM2 := matrix.OnesLike(forward_info["M2"])
//
// 	dLdM2 := dLdP.Multiply(dPdM2)
//
// 	dPdB2 := matrix.OnesLike(weights["B2"])
//
// 	dLdB2 := dLdP.Multiply(dPdB2).AxisSum(0)
//
// 	dM2dW2 := forward_info["O1"].Transpose()
//
// 	dLdW2 := dM2dW2.Dot(dLdP)
//
// 	dM2dO1 := weights["W2"].Transpose()
//
// 	dLdO1 := dLdM2.Dot(dM2dO1)
//
// 	dO1dN1 := forward_info["N1"].Apply(func(f float64) float64 { return sigmoid(f) }).Multiply(forward_info["N1"].Apply(func(f float64) float64 { return 1 - sigmoid(f) }))
//
// 	dLdN1 := dLdO1.Multiply(dO1dN1)
//
// 	dN1dB1 := matrix.OnesLike(weights["B1"])
//
// 	dN1dM1 := matrix.OnesLike(forward_info["M1"])
//
// 	dLdB1 := dLdN1.Multiply(dN1dB1).AxisSum(0)
//
// 	dLdM1 := dLdN1.Multiply(dN1dM1)
//
// 	dM1dW1 := forward_info["W"].Transpose()
//
// 	dLdW1 := dM1dW1.Dot(dLdM1)
//
// 	_loss_gradients := WF{
// 		"W2": dLdW2,
// 		"B2": dLdB2.AxisSum(0),
// 		"W1": dLdW1,
// 		"B1": dLdB1.AxisSum(0),
// 	}
//
// 	return _loss_gradients
// }
//
// func Predict(X *matrix.Matrix, weights WF) *matrix.Matrix {
// 	M1 := X.Dot(weights["W1"])
//
// 	N1 := M1.Add(weights["B1"])
//
// 	O1 := N1.Apply(func(f float64) float64 { return sigmoid(f) })
//
// 	M2 := O1.Dot(weights["W2"])
//
// 	P := M2.Add(weights["B2"])
//
// 	return P
// }
