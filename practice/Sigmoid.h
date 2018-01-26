#ifndef SIGMOID_H
#define SIGMOID_H

#include"Matrix.h"

class Sigmoid{
public:
	Sigmoid();
	~Sigmoid();

	Matrix* PropagateForward(const Matrix& in, Matrix* out);
	Matrix* PropagateBackward(const Matrix& in, Matrix* out);

private:
	Matrix m_y;
};
#endif