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
	X, y *matrix.Matrix
	size int
}

func Forward_Linear_Regression(X, y *matrix.Matrix, weights WF) (float32, WF) {
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

	loss := y.Subtract(P).Apply(func(f float32) float32 { return f * f }).Mean()

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

// func (l *LinearReg) Batch_Fit(d *Dataset, num_epochs int, lr float32, batch_size int) *LinearReg {
// 	batches := d.X.Rows / batch_size
// 	num_epochs /= batches
//
// 	for i := range batches {
// 		X, y := d.X.Data[i*batch_size:(i+1)*batch_size-1], d.Y.Data[i*batch_size:(i+1)*batch_size-1]
// 		l = l.Fit(matrix.InitMatrix(X), matrix.InitMatrix(y), num_epochs, lr)
// 	}
// 	return l
// }

func (l *LinearReg) Fit(d *Dataset, num_epochs int, lr float32) *LinearReg {
	weights := WF{
		"W": l.W,
		"B": l.B,
	}
	for range num_epochs {
		_, forward_info := Forward_Linear_Regression(d.X, d.Y, weights)

		loss_gradients := Loss_Gradients(forward_info, weights)

		for key := range weights {
			weights[key] = weights[key].Subtract(loss_gradients[key].SingleMul(lr))
		}
	}

	return &LinearReg{
		W: weights["W"],
		B: weights["B"],
	}
}

func (l *LinearReg) RMSE(X, y *matrix.Matrix) float32 {
	if X.Rows != y.Rows {
		panic("X and y must have same number of rows")
	}

	predicted := l.Predict(X)
	diff := predicted.Subtract(y)
	sumSqauredError := diff.Apply(func(f float32) float32 { return f * f }).Mean()

	return float32(math.Sqrt(float64(sumSqauredError)))
}

func init_weights(input_size, hidden_size int, g *rand.Rand) WF {
	weights := WF{
		"W1": matrix.Random(input_size, hidden_size, g),
		"B1": matrix.Random(1, hidden_size, g),
		"W2": matrix.Random(hidden_size, 1, g),
		"B2": matrix.Random(1, 1, g),
	}

	return weights
}

// func (d *Dataset)Generate_Batch(start, batch_size int) *Batch {
// 	if start+batch_size > d.X.Rows {
// 		batch_size = d.X.Rows - start
// 	}
//
// 	X, y := d.X.Data[]
// }

// func Forward_Loss(X, y *matrix.Matrix, weights WF) {
// 	M1 := X.Dot(weights["W1"])
//
// 	N1 := M1.Add(weights["B1"])
//
// 	O1 := N1.Apply(func(f float32) float32 { return sigmoid(f) })
//
// }

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-x))))
}
