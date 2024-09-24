package main

import (
	"fmt"
	"github.com/raunakwete43/deepgo.git/lib/linear_reg"
	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

func main() {
	X := matrix.InitMatrix([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	y := matrix.InitMatrix([][]float64{
		{6},
		{15},
	})

	// Example weights
	weights := linearreg.WF{
		"W": matrix.InitMatrix([][]float64{
			{0.5},
			{0.5},
			{0.5},
		}),
		"B": matrix.InitMatrix([]float64{
			0.1,
		}),
	}

	for range 100 {
		// Perform forward linear regression
		_, forward_info := linearreg.Forward_Linear_Regression(X, y, weights)

		// Print results

		loss_gradients := linearreg.Loss_Gradients(forward_info, weights)

		for key := range weights {
			weights[key] = weights[key].Subtract(loss_gradients[key].SingleMul(0.002))
		}
	}

	fmt.Println("Weights -> ", weights["W"].Value())
	fmt.Println("Bias -> ", weights["B"].Value())
	fmt.Println("Predictions -> ",
		X.Dot(weights["W"]).Add(weights["B"]).Value())

	fmt.Println("Predicted Values => ", linearreg.Predict(X, weights).Value())
}
