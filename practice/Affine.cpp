#include "Affine.h"
#include"UpdatingStyleBase.h"

Affine::Affine(const Matrix& w, const Matrix& b,const double learningRate, UpdatingStyleFactory::UPDATING_STYLE optimizer_ID) :
	m_w(w),
	m_b(b)
{
	m_pOptimizer = UpdatingStyleFactory::GetInstance().Create(optimizer_ID, learningRate);
}

Affine::~Affine()
{
}

Matrix* Affine::PropagateForward(const Matrix& in_x, Matrix* out) {

	m_x = in_x;
	*out = m_x*m_w;
	*out += m_b;

	return out;
}

Matrix* Affine::PropagateBackward(const Matrix& in, Matrix* out) {

	Matrix tmp;
	*out = in * (*(m_w.Transpose(&tmp)));

	//�o�C�A�X�̍X�V
	m_pOptimizer->Update(&m_b, in);

	//�d�݃p�����[�^�̍X�V
	Matrix grads = (*(m_x.Transpose(&tmp))*in);
	grads.SetSize();
	m_pOptimizer->Update(&m_w, grads);

	return out;
}