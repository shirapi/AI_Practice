#include "UpdatingStyleFactory.h"
#include"UpdatingStyleBase.h"
#include"SGD.h"

UpdatingStyleFactory* UpdatingStyleFactory::m_pInstance = nullptr;

UpdatingStyleFactory::UpdatingStyleFactory()
{
}

UpdatingStyleFactory::~UpdatingStyleFactory()
{
}

UpdatingStyleBase* UpdatingStyleFactory::Create(UPDATING_STYLE id, double learningRate) {

	UpdatingStyleBase* ret = nullptr;

	switch (id) {
	case UPDATING_STYLE::SGD_STYLE:
		ret = new SGD(learningRate);
		break;
	}

	return ret;
}