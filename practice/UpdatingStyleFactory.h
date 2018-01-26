#ifndef UPDATINGSTYLEFACTORY_H
#define UPDATINGSTYLEFACTORY_H

class UpdatingStyleBase;

class UpdatingStyleFactory{
public:
	static UpdatingStyleFactory& GetInstance() {
		if (m_pInstance == nullptr) {
			m_pInstance = new UpdatingStyleFactory();
		}
		return *m_pInstance;
	}

	enum UPDATING_STYLE {
		SGD_STYLE,
		MOMENTUM,
		ADAGRAD
	};

	UpdatingStyleBase* Create(UPDATING_STYLE id, double learningRate);

private:
	static UpdatingStyleFactory* m_pInstance;
	UpdatingStyleFactory();
	~UpdatingStyleFactory();
};
#endif