package matrix

func (m1 *Matrix) subtract(m2 *Matrix) *Matrix {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		panic("Matrix dimensions do not match for addition")
	}

	data1 := m1.Data
	data2 := m2.Data
	result := make([][]float64, len(data1))
	for i, row := range data1 {
		result[i] = make([]float64, len(row))
		for j := range row {
			result[i][j] = data1[i][j] - data2[i][j]
		}
	}
	return InitMatrix(result)

}

func (m *Matrix) singleSub(a float64) *Matrix {
	result := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		result[i] = make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			result[i][j] = m.Data[i][j] - a
		}
	}
	return InitMatrix(result)
}

func (m1 *Matrix) sub_with_lead_dim(m2 *Matrix) *Matrix {
	result := make([][]float64, m1.Rows)
	for i, row := range m1.Data {
		result[i] = make([]float64, m1.Cols)
		for j := range row {
			result[i][j] = row[j] - m2.Data[0][j]
		}
	}
	return InitMatrix(result)
}
