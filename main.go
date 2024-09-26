package main

import (
	"fmt"
	"math/rand"

	"github.com/raunakwete43/deepgo.git/lib/linear_reg"
)

func main() {
	g := rand.New(rand.NewSource(1234))

	data := linearreg.Load_CSV("./data.csv")
	data.Shuffle(g)
	model := linearreg.Init_Linear(data, g)

	fmt.Println(data.X.Shape())
	fmt.Println(model.W.Shape())

	model = model.Fit(data, 500, 1e-6)

	data.Shuffle(g)
	fmt.Println(model.RMSE(data.X, data.Y))
}
