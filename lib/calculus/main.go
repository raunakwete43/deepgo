package calculus

import (
	"github.com/raunakwete43/deepgo.git/lib/matrix"
)

type Array_Function func(m *matrix.Matrix) *matrix.Matrix
type Chain []Array_Function

func Derv(f Array_Function, m *matrix.Matrix) *matrix.Matrix {
	h := float32(1e-4)
	x := f(m.Add(h))      // f(a+h)
	y := f(m.Subtract(h)) // f(a-h)
	return x.Subtract(y).SingleDiv(2 * h)
}

func ChainDerv2(chain Chain, m *matrix.Matrix) *matrix.Matrix {
	if len(chain) == 2 && (m.Rows == 1 || m.Cols == 1) {
		f1 := chain[0]
		f2 := chain[1]

		f1_of_x := f1(m)

		// df1/dx
		df1dx := Derv(f1, m)

		// df2/du(f1(x))
		df2du := Derv(f2, f1_of_x)

		return df1dx.Multiply(df2du)
	}

	return nil
}

func ChainDerv3(chain Chain, m *matrix.Matrix) *matrix.Matrix {
	if len(chain) == 3 && (m.Rows == 1 || m.Cols == 1) {
		f1 := chain[0]
		f2 := chain[1]
		f3 := chain[2]

		f1_of_x := f1(m)
		f2_of_x := f2(f1_of_x)

		df3du := Derv(f3, f2_of_x)

		df2du := Derv(f2, f1_of_x)

		// df1/dx
		df1dx := Derv(f1, m)

		return df1dx.Multiply(df2du).Multiply(df3du)
	}

	return nil
}

func matmul_forward(x, w *matrix.Matrix) *matrix.Matrix {
	if x.Cols == w.Rows {
		n := x.Dot(w)
		return n
	}
	return nil
}

func matmul_backward_first(_, w *matrix.Matrix) *matrix.Matrix {
	dNdX := w.Transpose()
	return dNdX
}

func matrix_forward_extra(x, w *matrix.Matrix, f Array_Function) *matrix.Matrix {
	if x.Cols == w.Rows {
		n := x.Dot(w)
		return f(n)
	}
	return nil
}

func matrix_Function_Backward_1(x, w *matrix.Matrix, sigma Array_Function) *matrix.Matrix {
	if x.Cols == w.Rows {
		n := x.Dot(w)

		dsdn := Derv(sigma, n)

		dndx := w.Transpose()

		return dsdn.Dot(dndx)
	}
	return nil
}

func Matrix_Function_Forward_Sum(x, w *matrix.Matrix, sigma Array_Function) float32 {
	if x.Cols == w.Rows {
		n := x.Dot(w)

		s := sigma(n)

		l := s.Sum()

		return l
	}

	panic("Error while calculating forward pass")
}

func Matrix_Function_Backward_Sum_1(x, w *matrix.Matrix, sigma Array_Function) *matrix.Matrix {
	if x.Cols == w.Rows {
		n := x.Dot(w)
		// s := sigma(n)

		// Sum of all elemets
		// l := s.Sum()

		// dlds
		// dlds := matrix.OnesLike(s)

		// dsdn
		dsdn := Derv(sigma, n)

		// dldn
		// dldn := dlds.Multiply(dsdn)

		// dndx
		dndx := w.Transpose()

		// dldx
		dldx := dsdn.Dot(dndx)

		return dldx
	}
	return nil
}
