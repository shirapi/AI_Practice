#ifndef AFFINE_H
#define AFFINE_H

#include"Matrix.h"
#include"UpdatingStyleFactory.h"

class UpdatingStyleBase;

class Affine{
public:
	Affine(const Matrix& w, const Matrix& b,const double learningRate, UpdatingStyleFactory::UPDATING_STYLE optimizer_ID);
	//ダミーコンストラクタ
	Affine(){};
	~Affine();

	Matrix* PropagateForward(const Matrix& in_x, Matrix* out);
	Matrix* PropagateBackward(const Matrix& in, Matrix* out);

private:
	Matrix m_x;
	Matrix m_w;
	Matrix m_b;
	UpdatingStyleBase* m_pOptimizer;
};
#endif