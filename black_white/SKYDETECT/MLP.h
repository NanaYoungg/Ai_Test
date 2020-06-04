#pragma once
#define MAXLAYER	10

#define LEARNING_RATE 0.1

class CMLP
{
public:
	CMLP();
	~CMLP();
	int m_iNumInNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;	// hidden only
	int m_iNumTotalLayer;	// inputlayer+hiddenlayer+outputlayer
	int m_iNumNodes[MAXLAYER];	// [0]-input layer,[1...]-hidden layer,[m_iNumHiddenLayer+1]-output layer
	double** m_Weight[MAXLAYER];	// [시작layer번호][시작노드번호][연결노드번호]
	double** m_NodeOut;				// [layer][node]

	double *pInValue, *pOutValue;
	double *pCorrectOutValue;

	double** m_ErrorGrident;		// [layer][node]


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
