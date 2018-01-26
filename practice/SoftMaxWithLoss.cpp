#include "SoftMaxWithLoss.h"

SoftMaxWithLoss::SoftMaxWithLoss(){
}

SoftMaxWithLoss::~SoftMaxWithLoss(){
}

double SoftMaxWithLoss::Forward(const Matrix& in, const Matrix& t) {

	m_y = m_x = in;
	m_t = t;

	//SoftMax
	//�I�[�o�[�t���[�΍�
	double c = m_x.Max();

	//�w���֐�
	for (unsigned int i = 0; i < m_x.m_Row; ++i) {
		for (unsigned int j = 0; j < m_x.m_Column; ++j) {
			m_y.m_Mat[i][j] = exp(m_x.m_Mat[i][j] - c);
		}
	}

	//�w���֐��̘a
	double sum = 0.0;
	for (unsigned int i = 0; i < m_x.m_Row; ++i) {
		for (unsigned int j = 0; j < m_x.m_Column; ++j) {
			sum += m_y.m_Mat[i][j];
		}
	}

	m_y *= 1 / sum;

	//CrossEntropyError
	//log0�̎��̃}�C�i�X������ɂȂ�̂�h�~
	double delta = 1e-7;

	sum = 0.0;

	for (unsigned int i = 0; i < m_y.m_Row; ++i) {
		for (unsigned int j = 0; j < m_t.m_Column; ++j) {
			sum += (m_t.m_Mat[i][j] * log(m_y.m_Mat[i][j] + delta));
		}
	}

	return -sum;
}

Matrix* SoftMaxWithLoss::Backword(Matrix* out) {

	out->m_Mat = vector<vector<double>>(m_x.m_Row, vector<double>(m_x.m_Column));
	out->SetSize();

	for (unsigned int i = 0; i < m_x.m_Row; ++i) {
		for (unsigned int j = 0; j < m_x.m_Column; ++j) {
			out->m_Mat[i][j] = m_y.m_Mat[i][j] - m_t.m_Mat[i][j];
		}
	}

	return out;
}