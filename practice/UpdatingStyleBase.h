#ifndef UPDATINGSTYLEBASE_H
#define UPDATINGSTYLEBASE_H

#include"Matrix.h"

class UpdatingStyleBase{
public:
	UpdatingStyleBase(double learningRate) :
		kLearningRate(learningRate) {};

	virtual ~UpdatingStyleBase() {};

	virtual void Update(Matrix* x, const Matrix& grads) = 0;

protected:
	const double kLearningRate;
};
#endif