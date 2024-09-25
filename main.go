package main

import (
	"fmt"
	"math/rand"

	"github.com/raunakwete43/deepgo.git/lib/linear_reg"
	// "github.com/raunakwete43/deepgo.git/lib/matrix"
)

func main() {
	g := rand.New(rand.NewSource(1234))
	data := linearreg.Load_CSV("./data.csv")
	model := linearreg.Init_Linear(data, g)

	fmt.Println(data.X.Shape())
	fmt.Println(model.W.Shape())

	model = model.Fit(data, 1000, 1e-4)

	fmt.Println(*model.W, model.B.Value())
	fmt.Println(model.RMSE(data.X, data.Y))

}
