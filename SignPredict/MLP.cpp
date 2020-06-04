#include "MLP.h"
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>


CMLP::CMLP()
	: m_iNumInNodes(0)
	, m_iNumOutNodes(0)
{
	int i;

	m_NodeOut = NULL;
	for (i = 0; i < MAXLAYER; i++)
		m_Weight[i] = NULL;

	m_ErrorGrident = NULL;
}


CMLP::~CMLP()
{
	int i,j,k;
	if (m_NodeOut != NULL)
	{
		for(i = 0;i<(m_iNumTotalLayer + 1);i++)
			free(m_NodeOut[i]);
		free(m_NodeOut);
	}

	for (k = 0; k < MAXLAYER; k++)
	{
		if (m_Weight[k] != NULL)
		{
			for (j = 0; j < m_iNumNodes[k] + 1; j++)
				free(m_Weight[k][j]);
			free(m_Weight[k]);
		}
	}

	if (m_ErrorGrident != NULL)
	{
		for (i = 0; i<(m_iNumTotalLayer); i++)
			free(m_ErrorGrident[i]);
		free(m_ErrorGrident);
	}

}


bool CMLP::Create(int InNode, int OutNode, int hiddenlayer, int* pHiddenNode)
{
	int layer, node;

	m_iNumInNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = hiddenlayer;
	m_iNumTotalLayer = m_iNumHiddenLayer + 2;

	m_iNumNodes[0] = m_iNumInNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_iNumNodes[layer + 1] = pHiddenNode[layer];
	m_iNumNodes[m_iNumHiddenLayer + 1] = m_iNumOutNodes;

	// 각노드별 출력메모리할당=[layerno][nodeno]
	// 입력:m_NodeOut[0][],출력:m_NodeOut[m_iNumTotalLayer-1][]
	// 정답:m_NodeOut[m_iNumTotalLayer][]
	m_NodeOut = (double **)malloc((m_iNumTotalLayer+1) * sizeof(double*));	// 정답(+1)
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double *)malloc(m_iNumNodes[layer] * sizeof(double));
	m_NodeOut[layer] = (double *)malloc(m_iNumOutNodes * sizeof(double));	// 정답

	// 가중치 메모리할당 m_Weight[시작layer번호][시작노드번호][연결노드번호]
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_iNumNodes[layer]+1) * sizeof(double*)); //바이어스(+1)
		for (node = 0; node < m_iNumNodes[layer] + 1; node++)
			m_Weight[layer][node] = (double*)malloc(m_iNumNodes[layer + 1] * sizeof(double));
	}

	pInValue = m_NodeOut[0];
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

	initw();

	return true;
}


void CMLP::initw()
{
	int layer, snode, enode;

	srand(time(NULL));

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (snode = 0; snode < m_iNumNodes[layer] + 1; snode++)	// for 바이어스를 위해 +1
		{
			for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
			{
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;	// -0.5~0.5
			}
		}
	}
}


void CMLP::forward()
{
	int layer, snode, enode;
	double wsum;

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
		{
			wsum = 0.0;
			for (snode = 0; snode < m_iNumNodes[layer]; snode++)
			{
				wsum += (m_Weight[layer][snode][enode] * m_NodeOut[layer][snode]);
			}
			wsum += (m_Weight[layer][snode][enode] * 1);	// 바이어스

			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}

}


double CMLP::ActivationFunc(double u)
{
	// step func
//	if (u > 0)	return 1.0;
//	else		return 0.0;

	return 1 / (1 + exp(-u));
}


void CMLP::BackPropagationLearning()
{
	int layer, snode, enode;

	if (m_ErrorGrident == NULL)
	{
		// 각노드별 출력메모리할당=[layerno][nodeno]
		// 입력:m_ErrorGrident[0][],출력:m_ErrorGrident[m_iNumTotalLayer-1][]
		m_ErrorGrident = (double **)malloc((m_iNumTotalLayer) * sizeof(double*));	// 
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGrident[layer] = (double *)malloc(m_iNumNodes[layer] * sizeof(double));
	}

	// 출력층 에러경사
	for (snode = 0; snode < m_iNumNodes[m_iNumTotalLayer - 1]; snode++)
	{
		m_ErrorGrident[m_iNumTotalLayer - 1][snode] =
			(pCorrectOutValue[snode] - pOutValue[snode])
			*(pOutValue[snode])*(1 - pOutValue[snode]);
	}

	//에러경사값을 계산
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (snode = 0; snode < m_iNumNodes[layer]; snode++)
		{
			m_ErrorGrident[layer][snode] = 0.0;
			for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
			{
				m_ErrorGrident[layer][snode] += m_ErrorGrident[layer + 1][enode] * m_Weight[layer][snode][enode];
			}
			m_ErrorGrident[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
		}
	}

	// 가중치갱신
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode < m_iNumNodes[layer]; snode++)
			{
				m_Weight[layer][snode][enode] += (LEARNING_RATE*m_ErrorGrident[layer + 1][enode] * m_NodeOut[layer][snode]);
			}
			m_Weight[layer][snode][enode] += (LEARNING_RATE*m_ErrorGrident[layer + 1][enode] * 1);	// 바이어스
		}	
	}


}


bool CMLP::SaveWeight(char* fname)
{
	FILE* fp;

	if ((fp = fopen(fname, "wt")) == NULL)
		return false;

	int layer, snode, enode;
	// innode outnode, hlayer
	// node_layer0 node_layer1 .......
	fprintf(fp, "%d %d %d\n", m_iNumInNodes, m_iNumOutNodes, m_iNumHiddenLayer);
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fprintf(fp, "%d ", m_iNumNodes[layer]);
	}
	fprintf(fp,"\n");

	// save weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode < m_iNumNodes[layer]; snode++)
			{
				fprintf(fp,"%.9lf ", m_Weight[layer][snode][enode]);
			}
			fprintf(fp, "%.9lf ", m_Weight[layer][snode][enode]);	// 바이어스	
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	return true;
}


bool CMLP::LoadWeight(char* fname)
{
	FILE* fp;

	if ((fp = fopen(fname, "rt")) == NULL)
		return false;

	int layer, snode, enode;
	// innode outnode, hlayer
	// node_layer0 node_layer1 .......
	fscanf(fp, "%d %d %d", &m_iNumInNodes, &m_iNumOutNodes, &m_iNumHiddenLayer);
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fscanf(fp, "%d ", &m_iNumNodes[layer]);
	}

	// load weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 0; enode < m_iNumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode < m_iNumNodes[layer]; snode++)
			{
				fscanf(fp, "%lf ", &m_Weight[layer][snode][enode]);
			}
			fscanf(fp, "%lf ", &m_Weight[layer][snode][enode]);	// 바이어스	
		}
	}

	fclose(fp);

	return true;
}
