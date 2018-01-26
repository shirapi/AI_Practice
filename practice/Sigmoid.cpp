#include "Sigmoid.h"

Sigmoid::Sigmoid(){
}

Sigmoid::~Sigmoid(){
}

Matrix* Sigmoid::PropagateForward(const Matrix& in, Matrix* out) {

	*out = in;

	for (unsigned int i = 0; i < in.m_Row; ++i) {
		for (unsigned int j = 0; j < in.m_Column; ++j) {
			out->m_Mat[i][j] = 1.0 / (1.0 + exp(-(in.m_Mat[i][j])));
		}
	}

	m_y = *out;

	return out;
}

Matrix* Sigmoid::PropagateBackward(const Matrix& in, Matrix* out) {

	out->m_Mat = vector<vector<double>>(in.m_Row, vector<double>(in.m_Column));
	out->SetSize();

	for (unsigned int i = 0; i < in.m_Row; ++i) {
		for (unsigned int j = 0; j <in.m_Column; ++j) {
			out->m_Mat[i][j] = in.m_Mat[i][j] * m_y.m_Mat[i][j] * (1.0 - m_y.m_Mat[i][j]);
		}
	}

	return out;
}