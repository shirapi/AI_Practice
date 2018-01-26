#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include"Matrix.h"
#include<vector>

namespace myMath {
	//活性化関数
	//入力値が０を超えていたら１、それ以外は０を出力
	double Step(const double& x);
	Matrix* Step(const Matrix& x, Matrix* s);

	//シグモイド関数 1/(1+exp(-x)) ...exp(-x)はe^-x
	class SigmoidLayer {

	};
	double Sigmoid(const double& x);
	Matrix* Sigmoid(const Matrix& x, Matrix* s);

	//入力値が０を超えていたらそのまま、それ以外は０を出力
	double Relu(const double& x);
	Matrix* Relu(const Matrix& x, Matrix* s);

	//出力層の設定用関数
	//ソフトマックス関数 yk=exp(ak)/Σexp(ai)
	Matrix* SoftMax(const Matrix& x, Matrix* s);

	//損失関数
	//２乗和誤差関数 1/2Σ(yk-tk)^2
	//						出力データ			教師データ
	double SumSquaresError(const Matrix& y, const Matrix& t);

	//交差エントロピー誤差関数 -Σtk log yk
	//						出力データ			教師データ
	double CrossEntropyError(const Matrix& y, const Matrix& t);
}
#endif