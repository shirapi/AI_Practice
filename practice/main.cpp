#include "DataSet.h"
#include"Matrix.h"
#include"ActivationFunction.h"
#include<time.h>
#include<new>

#include"SoftMax.h"
#include"CrossEntropyError.h"
#include"SoftMaxWithLoss.h"
#include"Sigmoid.h"
#include"Affine.h"

//min<= random <= max
int Random(int min, int max)
{
	static bool seedrandom = false;

	if (seedrandom == false) {
		srand((unsigned int)time(NULL));
		seedrandom = true;
	}

	return min + rand() % (max - min + 1);
}

double func(const std::vector<double>& x) {
	return (0.01*(x[0]*x[0])) + 0.1*x[0];
}

double func2(const std::vector<double>& x) {
	return x[0]*x[0]+x[1]*x[1];
}

//vector<vector<double>>* DifferentiateLossFunc(double(*pFunc)(const Matrix&, const Matrix&), const Matrix& x, const Matrix& t, vector<vector<double>>* grad) {
//	double h = 1e-4;
//
//	for (int i = 0; i < x.m_Row; ++i) {
//		for (int j = 0; j < x.m_Column; ++j) {
//			Matrix tmp1 = x;
//			Matrix tmp2 = x;
//
//			tmp1.m_Mat[i][j] += h;
//			tmp2.m_Mat[i][j] -= h;
//
//			(*grad)[i][j] = (pFunc(tmp1, t) - pFunc(tmp2, t)) / (2 * h);
//		}
//	}
//}

double Differentiate(double(*pFunc)(const std::vector<double>&), const std::vector<double>& x, int index) {
	double h = 1e-4;

	std::vector<double> tmp1 = x;
	std::vector<double> tmp2 = x;

	tmp1[index] += h;
	tmp2[index] -= h;

	return (pFunc(tmp1) - pFunc(tmp2)) / (2 * h);
}

void main(void) {

	Mnist mnist;
	vector<vector<double>> imgs= vector<vector<double>>(28, vector<double>(28));
	vector<double> labels = vector<double>(60000);
	Mnist::ImageFileInfo picFileInfo;
	Mnist::ImageFileInfo labelFileInfo;
	mnist.ReadTrainingFile("MnistData\\train-images.idx3-ubyte", &imgs, &picFileInfo);
	mnist.ReadLabelFile("MnistData\\train-labels.idx1-ubyte", &labels, &labelFileInfo);

	//�n�C�p�[�p�����[�^
	const int kLearningNum = 10;//�P��������̌��z���X�V�����
	const int kBatchSize = 1000;//�w�K����摜����
	const int kZNum = 100;//���ԑw�̃j���[�����̐�
	const int kYNum = 10;//�ŏI�o�̓j���[������
	const double learningRate = 0.01;//�w�K��

	const int kAffineLayerMax = 2;
	//�d���p�����[�^
	vector<vector<double>> mat_w[kAffineLayerMax];
	Matrix w[kAffineLayerMax];

	mat_w[0] = vector<vector<double>>(picFileInfo.rows*picFileInfo.columns, vector<double>(kZNum, 0.1));
	w[0] = mat_w[0];
	w[0].SetSize();
	w[0].InitGaussDistribution(0.0, 0.01);

	mat_w[1] = vector<vector<double>>(kZNum, vector<double>(kYNum, 0.1));
	w[1] = mat_w[1];
	w[1].SetSize();
	w[1].InitGaussDistribution(0.0, 0.01);

	//�o�C�A�X�p�����[�^
	vector<vector<double>> mat_b[kAffineLayerMax];
	Matrix b[kAffineLayerMax];

	mat_b[0] = vector<vector<double>>(1, vector<double>(w[0].m_Column, 0.1));
	b[0] = mat_b[0];
	b[0].SetSize();
	b[0].InitGaussDistribution(0.0, 0.01);

	mat_b[1] = vector<vector<double>>(1, vector<double>(w[1].m_Column, 0.1));
	b[1] = mat_b[1];
	b[1].SetSize();
	b[1].InitGaussDistribution(0.0, 0.01);

	Affine affine[kAffineLayerMax];
	for (int i = 0; i < kAffineLayerMax; ++i) {
		new(affine + i) Affine(w[i], b[i], learningRate, UpdatingStyleFactory::UPDATING_STYLE::SGD_STYLE);
	}

	for (int batch = 0; batch < kBatchSize; ++batch) {

		//�f�[�^�̌���
		int index = Random(0, picFileInfo.numberOfImages - 1);

		if (batch > 900) {
			int a = 0;
		}

		for (int learn = 0; learn < kLearningNum; ++learn) {
			//Matrix�̐���
			//���̓p�����[�^
			vector<vector<double>> mat_x;
			mat_x.push_back(imgs[index]);
			Matrix x = mat_x;

			//���ԑw�j���[�����ɏo��
			Matrix y;
			affine[0].PropagateForward(x, &y);

			//�������֐��ɂ�����
			Sigmoid sigmoid;
			sigmoid.PropagateForward(y, &y);

			//�ŏI�j���[�����ɏo��
			affine[1].PropagateForward(y, &y);

			//softmax�Ŋ����ɂ���
			SoftMax softmax;
			softmax.PropagateForward(y, &y);
			
			//�����֐��Ő����Ƃ̌덷���Z�o
			vector<vector<double>> mat_t = vector<vector<double>>(1, vector<double>(kYNum, 0.0));
			mat_t[0][labels[index]] = 1.0;
			Matrix t = mat_t;
			CrossEntropyError error;
			double loss = error.PropagateForward(y, t);
			printf("����\n0=%lf\t1=%lf\t2=%lf\t3=%lf\t4=%lf\t5=%lf\n6=%lf\t7=%lf\t8=%lf\t9=%lf\n", y.m_Mat[0][0], y.m_Mat[0][1], y.m_Mat[0][2], y.m_Mat[0][3], y.m_Mat[0][4], y.m_Mat[0][5], y.m_Mat[0][6], y.m_Mat[0][7], y.m_Mat[0][8], y.m_Mat[0][9]);
			printf("����\n0=%lf\t1=%lf\t2=%lf\t3=%lf\t4=%lf\t5=%lf\n6=%lf\t7=%lf\t8=%lf\t9=%lf\n", mat_t[0][0], mat_t[0][1], mat_t[0][2], mat_t[0][3], mat_t[0][4], mat_t[0][5], mat_t[0][6], mat_t[0][7], mat_t[0][8], mat_t[0][9]);
			printf("�덷��%lf\n", loss);

			//�t�`�d
			Matrix back;
			Matrix tmp;
			error.PropagateBackward(&tmp);
			back = tmp;
			softmax.PropagateBackward(back, &tmp);
			back = tmp;
			affine[1].PropagateBackward(back, &tmp);
			back = tmp;
			sigmoid.PropagateBackward(back, &tmp);
			back = tmp;
			affine[0].PropagateBackward(back, &tmp);
			back = tmp;
		}
	}

	while (1);
}