#include <stdio.h>

#include "MLP.h"    //�ڱⰡ ����� ""�� �߰�

CMLP MultiLayer; 


#define NUM_TRAINING_SET 4
#define NUM_INPUT 9
#define NUM_OUTPUT 4

#define MAX_EPOCH 1000000

int main()
{
	int n, p;
	int epoch;

	//�Ű�� ��Ʈ��ũ ����

	int hlayer[1] = { 2 };
	MultiLayer.Create(NUM_INPUT, NUM_OUTPUT, 1, hlayer);

	//�н������� �غ�

	double x[NUM_TRAINING_SET][NUM_INPUT] = {
						{ 1,1,1
						 ,0,0,1,
						  0,0,1 },  //��
						 { 1,0,0,
						   1,0,0,
						   1,1,1 }, //��
						  { 1,1,1
						   ,1,0,0,
							1,1,1 },//��
							{1,1,1,
							 1,0,1,
							 1,1,1} };  //��

												//������  //������  //������  //������
	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };   //���� (��ǥ��)


	if (MultiLayer.LoadWeight("..\\Weight\\weight.txt") == true)
	{
		printf("������ ����ġ�κ��� �н��� �����մϴ�.\n");

	}
	else
	{
		printf("���� ����ġ�κ��� ����ġ�κ��� ó������ �����մϴ�.\n");

	}

	//������
	for (n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (p = 0;p < NUM_INPUT;p++)
			MultiLayer.pInValue[p] = x[n][p];


		MultiLayer.forward();

		for (p = 0;p < NUM_INPUT;p++)
			printf("%.0f", MultiLayer.pInValue[p]);
		printf("=");

		for (p = 0;p < NUM_OUTPUT;p++)
			printf("%lf,", MultiLayer.pOutValue[p]);
		printf("\n");
	}
	getchar();

	// �н�
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		//�Է�,���� ����  -> ������ϱ�, �������н�
		for (n = 0; n < NUM_TRAINING_SET; n++)
		{
			//�Է� ����
			for (p = 0;p < NUM_INPUT;p++)
				MultiLayer.pInValue[p] = x[n][p];
			//���� ����
			for(p=0;p<NUM_OUTPUT;p++)
			MultiLayer.pCorrectOutValue[p] = d[n][p];


			//��°����
			MultiLayer.forward();


			for (p = 0;p<NUM_OUTPUT;p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p])*(MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);


			//�������н����� ����ġ����
			MultiLayer.BackPropagationLearning();
		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}

	MultiLayer.SaveWeight("..\\Weight\\weight.txt");

	
	//������
	for (n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (p = 0;p < NUM_INPUT;p++)
			MultiLayer.pInValue[p] = x[n][p];


		MultiLayer.forward();

		for (p = 0;p < NUM_INPUT;p++)
			printf("%.0f", MultiLayer.pInValue[p]);
		printf("=");

		for (p = 0;p < NUM_OUTPUT;p++)


			printf("%lf", MultiLayer.pOutValue[p]);
		printf("\n");
	
	}



	return 0;


}

