package linearreg

import (
	"encoding/csv"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type dataset struct {
	X [][]float64
	y []float64
}

type Dataset struct {
	X, Y   *matrix.Matrix
	N_Feat int
}

func load_data(filepath string) (*dataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return &dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return &dataset{}, err
	}

	var data dataset
	for _, record := range records {
		row := make([]float64, len(record)-1)
		for i := 0; i < len(record)-1; i++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[i]), 64)
			if err != nil {
				return &dataset{}, err
			}
			row[i] = float64(val)
		}
		data.X = append(data.X, row)

		val, err := strconv.ParseFloat(strings.TrimSpace(record[len(record)-1]), 64)
		if err != nil {
			return &dataset{}, err
		}
		data.y = append(data.y, float64(val))
	}

	return &data, nil
}

func Load_CSV(file_path string) *Dataset {
	d, err := load_data(file_path)
	if err != nil {
		return nil
	}

	return &Dataset{
		X:      matrix.InitMatrix(d.X),
		Y:      matrix.InitMatrix(d.y).Transpose(),
		N_Feat: len(d.X[0]),
	}
}

func (d *Dataset) Shuffle(g *rand.Rand) {
	if g != nil {
	}

	rand.Shuffle(len(d.X.Data), func(i, j int) {
		d.X.Data[i], d.X.Data[j] = d.X.Data[j], d.X.Data[i]
		d.Y.Data[i], d.Y.Data[j] = d.Y.Data[j], d.Y.Data[i]
	})
}

func (d *Dataset) Generate_Batch(start, batch_size int) *Batch {
	if start+batch_size > d.X.Rows {
		batch_size = d.X.Rows - start
	}

	X, y := d.X.Data[start:start+batch_size], d.Y.Data[start:start+batch_size]

	return &Batch{
		X:    matrix.InitMatrix(X),
		Y:    matrix.InitMatrix(y),
		Size: batch_size,
	}
}
