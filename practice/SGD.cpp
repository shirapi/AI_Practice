#include "SGD.h"

SGD::SGD(double learnengRate):
	UpdatingStyleBase(learnengRate)
{
}

SGD::~SGD(){

}

void SGD::Update(Matrix* x, const Matrix& grads) {
	*x -= (grads*kLearningRate);
}