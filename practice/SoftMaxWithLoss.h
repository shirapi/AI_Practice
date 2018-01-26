#ifndef SOFTMAXWITHLOSS_H
#define SOFTMAXWITHLOSS_H

#include"Matrix.h"

class SoftMaxWithLoss{
public:
	SoftMaxWithLoss();
	~SoftMaxWithLoss();

	double Forward(const Matrix& in, const Matrix& t);
	Matrix* Backword(Matrix* out);

private:
	Matrix m_x;
	Matrix m_t;
	Matrix m_y;
};
#endif