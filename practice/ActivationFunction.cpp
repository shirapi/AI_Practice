#include"ActivationFunction.h"

double myMath::Step(const double& x) {
	if (0 < x) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

Matrix* myMath::Step(const Matrix& x, Matrix* s) {
	*s = x;
	for (unsigned int i = 0; i < s->m_Row; ++i) {
		for (unsigned int j = 0; j < s->m_Column; ++j) {
			s->m_Mat[i][j] = Step(s->m_Mat[i][j]);
		}
	}
	return s;
}

double myMath::Sigmoid(const double& x) {
	return 1.0 / (1.0 + exp(-x));
}

Matrix* myMath::Sigmoid(const Matrix& x, Matrix* s) {
	*s = x;
	for (unsigned int i = 0; i < s->m_Row; ++i) {
		for (unsigned int j = 0; j < s->m_Column; ++j) {
			s->m_Mat[i][j] = Sigmoid(s->m_Mat[i][j]);
		}
	}
	return s;
}

double myMath::Relu(const double& x) {
	if (0 < x) {
		return x;
	}
	else {
		return 0.0;
	}
}

Matrix* myMath::Relu(const Matrix& x, Matrix* s) {
	*s = x;
	for (unsigned int i = 0; i < s->m_Row; ++i) {
		for (unsigned int j = 0; j < s->m_Column; ++j) {
			s->m_Mat[i][j] = Relu(s->m_Mat[i][j]);
		}
	}
	return s;
}

Matrix* myMath::SoftMax(const Matrix& x, Matrix* y) {
	*y = x;

	//オーバーフロー対策
	double c = y->Max();

	//指数関数
	for (unsigned int i = 0; i < y->m_Row; ++i) {
		for (unsigned int j = 0; j < y->m_Column; ++j) {
			y->m_Mat[i][j] = exp(y->m_Mat[i][j] - c);
		}
	}
	
	//指数関数の和
	double sum = 0;
	for (unsigned int i = 0; i < y->m_Row; ++i) {
		for (unsigned int j = 0; j < y->m_Column; ++j) {
			sum += y->m_Mat[i][j];
		}
	}

	*y *= 1/sum;
	return y;
}

double myMath::SumSquaresError(const Matrix& y, const Matrix& t) {
	double sum = 0.0;
	for (unsigned int i = 0; i < y.m_Row; ++i) {
		for (unsigned int j = 0; j < t.m_Column; ++j) {
			sum += (y.m_Mat[i][j] - t.m_Mat[i][j])*(y.m_Mat[i][j] - t.m_Mat[i][j]);
		}
	}
	return 0.5*sum;
}

double myMath::CrossEntropyError(const Matrix& y, const Matrix& t) {
	//log0の時のマイナス無限大になるのを防止
	double delta = 1e-7;

	double sum = 0.0;

	for (unsigned int i = 0; i < y.m_Row; ++i) {
		for (unsigned int j = 0; j < t.m_Column; ++j) {
			sum += (t.m_Mat[i][j] * log(y.m_Mat[i][j] + delta));
		}
	}

	return -sum;
}