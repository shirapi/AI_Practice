#ifndef SGD_H
#define SGD_H

#include "UpdatingStyleBase.h"

class SGD :public UpdatingStyleBase
{
public:
	SGD(double learningRate);
	virtual ~SGD();

	virtual void Update(Matrix* x, const Matrix& grads);
};
#endif