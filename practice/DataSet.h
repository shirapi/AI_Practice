#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <vector>

using std::string;
using std::vector;

class Mnist {
public:

	Mnist();
	~Mnist();

	struct ImageFileInfo {
		int magicNumber = 0;	//よくわからないもの
		int numberOfImages = 0;	//イメージ（画像数）
		unsigned int rows = 0;			//行数
		unsigned int columns = 0;		//列数
	};

	void ReadTrainingFile(string filename, vector<vector<double>>* pImages, ImageFileInfo* pFileInfo);
	void ReadLabelFile(string filename, vector<double>* pLabels, ImageFileInfo* pFileInfo);

private:
	double* m_pImages;
};

#endif