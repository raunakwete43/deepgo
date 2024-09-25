package linearreg

import (
	"encoding/csv"
	"os"
	"strconv"
	"strings"

	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type dataset struct {
	X [][]float32
	y []float32
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
		row := make([]float32, len(record)-1)
		for i := 0; i < len(record)-1; i++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[i]), 32)
			if err != nil {
				return &dataset{}, err
			}
			row[i] = float32(val)
		}
		data.X = append(data.X, row)

		val, err := strconv.ParseFloat(strings.TrimSpace(record[len(record)-1]), 32)
		if err != nil {
			return &dataset{}, err
		}
		data.y = append(data.y, float32(val))
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
