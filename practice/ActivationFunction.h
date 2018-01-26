#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include"Matrix.h"
#include<vector>

namespace myMath {
	//�������֐�
	//���͒l���O�𒴂��Ă�����P�A����ȊO�͂O���o��
	double Step(const double& x);
	Matrix* Step(const Matrix& x, Matrix* s);

	//�V�O���C�h�֐� 1/(1+exp(-x)) ...exp(-x)��e^-x
	class SigmoidLayer {

	};
	double Sigmoid(const double& x);
	Matrix* Sigmoid(const Matrix& x, Matrix* s);

	//���͒l���O�𒴂��Ă����炻�̂܂܁A����ȊO�͂O���o��
	double Relu(const double& x);
	Matrix* Relu(const Matrix& x, Matrix* s);

	//�o�͑w�̐ݒ�p�֐�
	//�\�t�g�}�b�N�X�֐� yk=exp(ak)/��exp(ai)
	Matrix* SoftMax(const Matrix& x, Matrix* s);

	//�����֐�
	//�Q��a�덷�֐� 1/2��(yk-tk)^2
	//						�o�̓f�[�^			���t�f�[�^
	double SumSquaresError(const Matrix& y, const Matrix& t);

	//�����G���g���s�[�덷�֐� -��tk log yk
	//						�o�̓f�[�^			���t�f�[�^
	double CrossEntropyError(const Matrix& y, const Matrix& t);
}
#endif