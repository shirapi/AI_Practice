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
		int magicNumber = 0;	//�悭�킩��Ȃ�����
		int numberOfImages = 0;	//�C���[�W�i�摜���j
		unsigned int rows = 0;			//�s��
		unsigned int columns = 0;		//��
	};

	void ReadTrainingFile(string filename, vector<vector<double>>* pImages, ImageFileInfo* pFileInfo);
	void ReadLabelFile(string filename, vector<double>* pLabels, ImageFileInfo* pFileInfo);

private:
	double* m_pImages;
};

#endif