#include"DataSet.h"

using std::ifstream;
using std::cout;
using std::endl;

Mnist::Mnist():
	m_pImages(nullptr)
{
}

Mnist::~Mnist() {
	delete[] m_pImages;
}

//バイト列からintへの変換
int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void Mnist::ReadTrainingFile(string filename, vector<vector<double>>* pImages, ImageFileInfo* pFileInfo) {

	ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);

	//ヘッダー部より情報を読取る。
	ifs.read((char*)&pFileInfo->magicNumber, sizeof(pFileInfo->magicNumber));
	pFileInfo->magicNumber = reverseInt(pFileInfo->magicNumber);
	ifs.read((char*)&pFileInfo->numberOfImages, sizeof(pFileInfo->numberOfImages));
	pFileInfo->numberOfImages = reverseInt(pFileInfo->numberOfImages);
	ifs.read((char*)&pFileInfo->rows, sizeof(pFileInfo->rows));
	pFileInfo->rows = reverseInt(pFileInfo->rows);
	ifs.read((char*)&pFileInfo->columns, sizeof(pFileInfo->columns));
	pFileInfo->columns = reverseInt(pFileInfo->columns);

	cout << pFileInfo->magicNumber << " " << pFileInfo->numberOfImages << " " << pFileInfo->rows << " " << pFileInfo->columns << endl;

	*pImages = vector<vector<double>>(pFileInfo->numberOfImages, vector<double>(pFileInfo->columns*pFileInfo->rows));

	for (int i = 0; i < pFileInfo->numberOfImages; i++) {

		for (unsigned int row = 0; row < pFileInfo->rows; row++) {
			for (unsigned int col = 0; col < pFileInfo->columns; col++) {
				unsigned char temp = 0;
				ifs.read((char*)&temp, sizeof(temp));
				(*pImages)[i][pFileInfo->rows*row + col] = (double)temp;
			}
		}
	}
}

void Mnist::ReadLabelFile(string filename, vector<double>* pLabels, ImageFileInfo* pFileInfo) {
	ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);

	//ヘッダー部より情報を読取る。
	ifs.read((char*)&pFileInfo->magicNumber, sizeof(pFileInfo->magicNumber));
	pFileInfo->magicNumber = reverseInt(pFileInfo->magicNumber);
	ifs.read((char*)&pFileInfo->numberOfImages, sizeof(pFileInfo->numberOfImages));
	pFileInfo->numberOfImages = reverseInt(pFileInfo->numberOfImages);

	cout << pFileInfo->numberOfImages << endl;

	*pLabels = vector<double>(pFileInfo->numberOfImages);

	for (int i = 0; i < pFileInfo->numberOfImages; i++) {
		unsigned char temp = 0;
		ifs.read((char*)&temp, sizeof(temp));
		(*pLabels)[i] = (double)temp;
		cout << (*pLabels)[i] << "\n" << endl;
	}
}