package nn

import (
	"fmt"

	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

func assert_same_shape(arr, arr_grad *matrix.Matrix) {
	if arr.Shape() != arr_grad.Shape() {
		fmt.Println("Arr Shape => ", arr.Shape())
		fmt.Println("Arr_Grad Shape => ", arr_grad.Shape())
		panic("Two arrays must be of same size.")
	}
}
