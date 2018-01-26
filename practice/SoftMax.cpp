#include "SoftMax.h"

SoftMax::SoftMax()
{
}

SoftMax::~SoftMax()
{
}

Matrix* SoftMax::PropagateForward(const Matrix& in, Matrix* out) {
	*out = m_x = in;

	//オーバーフロー対策
	double c = out->Max();

	//指数関数
	for (unsigned int i = 0; i < in.m_Row; ++i) {
		for (unsigned int j = 0; j < in.m_Column; ++j) {
			out->m_Mat[i][j] = exp(in.m_Mat[i][j] - c);
		}
	}

	//指数関数の和
	double sum = 0;
	for (unsigned int i = 0; i < in.m_Row; ++i) {
		for (unsigned int j = 0; j < in.m_Column; ++j) {
			sum += out->m_Mat[i][j];
		}
	}

	*out *= 1 / sum;
	return out;
}

Matrix* SoftMax::PropagateBackward(const Matrix& in, Matrix* out) {

	double sum = 0;
	for (unsigned int i = 0; i < m_x.m_Column; ++i) {
		sum += exp(m_x.m_Mat[0][i]);
	}

	double inDotSum = 0;
	for (unsigned int i = 0; i < m_x.m_Column; ++i) {
		inDotSum += in.m_Mat[0][i] * exp(m_x.m_Mat[0][i]);
	}

	out->m_Mat= vector<vector<double>>(1, vector<double>(m_x.m_Column));
	out->SetSize();

	for (unsigned int i = 0; i < m_x.m_Column; ++i) {
		out->m_Mat[0][i] = exp(m_x.m_Mat[0][i]) * ((-1.0 / (sum*sum) * inDotSum) + (in.m_Mat[0][i] / sum));
	}

	return out;
}