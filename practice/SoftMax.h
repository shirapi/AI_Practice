#ifndef SOFTMAX_H
#define SOFTMAX_H

#include"Matrix.h"

class SoftMax{
public:
	SoftMax();
	~SoftMax();

	Matrix* PropagateForward(const Matrix& in, Matrix* out);
	Matrix* PropagateBackward(const Matrix& in, Matrix* out);

private:
	Matrix m_x;
};
#endif