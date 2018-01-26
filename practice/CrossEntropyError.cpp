#include "CrossEntropyError.h"

CrossEntropyError::CrossEntropyError()
{
}

CrossEntropyError::~CrossEntropyError()
{
}

double CrossEntropyError::PropagateForward(const Matrix& in, const Matrix& t) {
	m_x = in;
	m_t = t;

	//log0の時のマイナス無限大になるのを防止
	double delta = 1e-7;

	double sum = 0.0;

	for (unsigned int i = 0; i < in.m_Row; ++i) {
		for (unsigned int j = 0; j < t.m_Column; ++j) {
			sum += (t.m_Mat[i][j] * log(in.m_Mat[i][j] + delta));
		}
	}

	return -sum;
}

Matrix* CrossEntropyError::PropagateBackward(Matrix* y) {

	y->m_Mat = vector<vector<double>>(1, vector<double>(m_x.m_Column));

	for (unsigned int i = 0; i < m_x.m_Column; ++i) {
		//詳しくは計算グラフを書いて計算する
		y->m_Mat[0][i] = -(m_t.m_Mat[0][i] / m_x.m_Mat[0][i]);
	}

	return y;
}