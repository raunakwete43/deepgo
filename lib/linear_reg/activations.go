package linearreg

import "math"

func sigmoid(x float64) float64 {
	return float64(1 / (1 + math.Exp(float64(-x))))
}
