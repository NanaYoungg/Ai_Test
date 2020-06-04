#include <stdio.h>

#include "MLP.h"
CMLP MultiLayer;

#define MAX_EPOCH	100000

int main()
{
	int layer, snode, enode;
	int n;
	int epoch;

	int hlayer[2] = { 2,3 };
	MultiLayer.Create(3, 1, 2, hlayer);

	double x[8][3] = { { 0,0,0 },{ 0,0,1 },{ 0,1,0 },{ 0,1,1 },
	{ 1,0,0 },{ 1,0,1 },{ 1,1,0 },{ 1,1,1 } };
	double d[8] = { 0,1,1,0,1,0,0,1 };

	MultiLayer.LoadWeight("weight.txt");

	for (n = 0; n < 8; n++)
	{
		MultiLayer.pInValue[0] = x[n][0];
		MultiLayer.pInValue[1] = x[n][1];
		MultiLayer.pInValue[2] = x[n][2];

		MultiLayer.forward();

		printf("%lf %lf %lf=%lf(%lf)\n", MultiLayer.pInValue[0], MultiLayer.pInValue[1], MultiLayer.pInValue[2],
			MultiLayer.pOutValue[0], d[n]);
	}

	// ÇÐ½À
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		for (n = 0; n < 8; n++)
		{
			MultiLayer.pInValue[0] = x[n][0];
			MultiLayer.pInValue[1] = x[n][1];
			MultiLayer.pInValue[2] = x[n][2];
			MultiLayer.pCorrectOutValue[0] = d[n];

			MultiLayer.forward();

			MSE += (MultiLayer.pCorrectOutValue[0] - MultiLayer.pOutValue[0])*(MultiLayer.pCorrectOutValue[0] - MultiLayer.pOutValue[0]);

			MultiLayer.BackPropagationLearning();
		}
		MSE /= 8;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}

	MultiLayer.SaveWeight("weight.txt");

	for (n = 0; n < 8; n++)
	{
		MultiLayer.pInValue[0] = x[n][0];
		MultiLayer.pInValue[1] = x[n][1];
		MultiLayer.pInValue[2] = x[n][2];

		MultiLayer.forward();

		printf("%lf %lf %lf=%lf(%lf)\n", MultiLayer.pInValue[0], MultiLayer.pInValue[1], MultiLayer.pInValue[2],
			MultiLayer.pOutValue[0], d[n]);
	}

	return 0;
}