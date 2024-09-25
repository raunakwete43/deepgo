package matrix

import "math/rand"

func (m *Matrix) Shuffle(g *rand.Rand) {
	if g != nil {
		n := len(m.Data)
		for i := n - 1; i > 0; i-- {
			j := g.Intn(i + 1)
			m.Data[i], m.Data[j] = m.Data[j], m.Data[i]
		}
	}

	rand.Shuffle(len(m.Data), func(i, j int) {
		m.Data[i], m.Data[j] = m.Data[j], m.Data[i]
	})
}
