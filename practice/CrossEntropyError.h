#ifndef CROSSENTROPYERROR_H
#define CROSSENTROPYERROR_H

#include"Matrix.h"

class CrossEntropyError {
public:
	CrossEntropyError();
	~CrossEntropyError();

	double PropagateForward(const Matrix& in, const Matrix& t);
	Matrix* PropagateBackward(Matrix* out);

private:
	Matrix m_x;
	Matrix m_t;
};
#endif