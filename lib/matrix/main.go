package matrix

import "math/rand"

type Matrix struct {
	Data [][]float64 // Data can be []float64 or [][]float64
	Rows int
	Cols int
}

func InitMatrix(data interface{}) *Matrix {
	switch d := data.(type) {
	case []float64:
		// Handle 1D matrix
		return &Matrix{
			Data: [][]float64{d},
			Rows: 1,
			Cols: len(d),
		}
	case [][]float64:
		// Handle 2D matrix
		return &Matrix{
			Data: d,
			Rows: len(d),
			Cols: len(d[0]),
		}
	default:
		// Handle unsupported types
		panic("Unsupported data type for matrix initialization")
	}
}

func (m *Matrix) Shape() [2]int {
	return [2]int{m.Rows, m.Cols}
}

func Arange(start, end, step float64) *Matrix {
	n := int((end-start)/step) + 1
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = start
		start += step
	}

	return InitMatrix(result)
}

func OnesLike(m *Matrix) *Matrix {
	return m.Apply(func(f float64) float64 { return 1 })
}

func Random(rows, cols int, generator *rand.Rand) *Matrix {
	result := make([][]float64, rows)
	if generator == nil {
		for i := 0; i < rows; i++ {
			result[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				result[i][j] = rand.Float64()
			}
		}
		return InitMatrix(result)
	}

	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = generator.Float64()
		}
	}

	return InitMatrix(result)
}

func (m *Matrix) Add(a interface{}) *Matrix {
	switch a.(type) {
	case float64:
		return m.singleAdd(a.(float64))
	case *Matrix:
		m2 := a.(*Matrix)
		if m2.Rows == 1 && m2.Cols == 1 {
			return m.singleAdd(m2.Data[0][0])
		}
		if m.Cols == m2.Cols && m2.Rows == 1 {
			return m.add_with_lead_dim(m2)
		}
		return m.add(m2)
	default:
		return nil
	}
}

func (m *Matrix) Subtract(a interface{}) *Matrix {
	switch a.(type) {
	case float64:
		return m.singleSub(a.(float64))
	case *Matrix:
		m2 := a.(*Matrix)
		if m2.Rows == 1 && m2.Cols == 1 {
			return m.singleSub(m2.Data[0][0])
		}
		if m.Cols == m2.Cols && m2.Rows == 1 {
			return m.sub_with_lead_dim(m2)
		}

		return m.subtract(m2)
	default:
		return nil
	}
}

func (m1 *Matrix) Multiply(m2 *Matrix) *Matrix {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		panic("Matrix dimensions do not match for Multiplication")
	}

	data1 := m1.Data
	data2 := m2.Data
	result := make([][]float64, len(data1))
	for i, row := range data1 {
		result[i] = make([]float64, len(row))
		for j := range row {
			result[i][j] = data1[i][j] * data2[i][j]
		}
	}
	return InitMatrix(result)

}

func (m1 *Matrix) Divide(m2 *Matrix) *Matrix {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		panic("Matrix dimensions do not match for Division")
	}

	data1 := m1.Data
	data2 := m2.Data
	result := make([][]float64, len(data1))
	for i, row := range data1 {
		result[i] = make([]float64, len(row))
		for j := range row {
			result[i][j] = data1[i][j] / data2[i][j]
		}
	}
	return InitMatrix(result)

}

func (m1 *Matrix) Dot(m2 *Matrix) *Matrix {
	if m1.Cols != m2.Rows {
		panic("Matrix dimensions are not compatible for multiplication")
	}

	data1 := m1.Data
	data2 := m2.Data
	result := make([][]float64, m1.Rows)

	for i := range data1 {
		result[i] = make([]float64, m2.Cols)
		for j := 0; j < m2.Cols; j++ {
			for k := 0; k < m1.Cols; k++ {
				result[i][j] += data1[i][k] * data2[k][j]
			}
		}
	}

	return InitMatrix(result)
}

func (m *Matrix) Transpose() *Matrix {
	result := make([][]float64, m.Cols)
	for i := 0; i < m.Cols; i++ {
		result[i] = make([]float64, m.Rows)
		for j := 0; j < m.Rows; j++ {
			result[i][j] = m.Data[j][i]
		}
	}
	return InitMatrix(result)
}

func (m *Matrix) Value() interface{} {
	if m.Rows == 1 {
		if m.Cols == 1 {
			return m.Data[0][0]
		}
		return m.Data[0]
	}

	return m.Data
}

func (m *Matrix) AxisSum(axis int) *Matrix {
	if axis == 0 {
		// Sum along rows (axis 0)
		sums := make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			for i := 0; i < m.Rows; i++ {
				sums[j] += m.Data[i][j]
			}
		}
		return InitMatrix(sums)
	} else if axis == 1 {
		// Sum along columns (axis 1)
		sums := make([]float64, m.Rows)
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				sums[i] += m.Data[i][j]
			}
		}
		return InitMatrix(sums)
	}
	return nil // Invalid axis
}

func (m *Matrix) Sum() float64 {
	rowSum := m.AxisSum(0)
	result := float64(0)
	for _, val := range rowSum.Data[0] {
		result += val
	}

	return result
}

func (m *Matrix) Mean() float64 {
	rowSum := m.AxisSum(0)
	result := float64(0)
	for _, val := range rowSum.Data[0] {
		result += val
	}

	return result / (float64(m.Rows) * float64(m.Cols))
}

func (m *Matrix) SingleMul(a float64) *Matrix {
	result := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		result[i] = make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			result[i][j] = m.Data[i][j] * a
		}
	}
	return InitMatrix(result)
}

func (m *Matrix) SingleDiv(a float64) *Matrix {
	result := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		result[i] = make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			result[i][j] = m.Data[i][j] / a
		}
	}
	return InitMatrix(result)
}

func (m *Matrix) Apply(f func(float64) float64) *Matrix {
	result := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		result[i] = make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			result[i][j] = f(m.Data[i][j])
		}
	}
	return InitMatrix(result)
}
