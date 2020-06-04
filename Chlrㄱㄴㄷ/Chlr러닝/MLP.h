#pragma once
#define MAXLAYER   10  

#define LEARNING_RATE 0.1  

class CMLP
{
public:
	CMLP();
	~CMLP();
	int m_iNumInNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;
	int m_iNumTotalLayer;
	int m_iNumNodes[MAXLAYER];
	double** m_Weight[MAXLAYER];
	double** m_NodeOut;

	double *pInValue, *pOutValue;
	double *pCorrectOutValue;

	double** m_ErrorGrident;


	bool Create(int InNode, int OutNode, int hiddenlayer, int* pHiddenNode);
private:
	void initw();
	double ActivationFunc(double u);
public:
	void forward();
	void BackPropagationLearning();
	bool SaveWeight(char* fname);
	bool LoadWeight(char* fname);
};

