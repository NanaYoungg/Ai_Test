#include <stdio.h>
#include <conio.h>

void DisplayMenu();

void Learning_Start();
void Save_Weight();
void Load_Weight();
void Learning_test();
void Test_Net();

#include "MLP.h"
CMLP MultiLayer;

#define NUM_TRAINING_SET	20
#define NUM_INPUT			3
#define NUM_OUTPUT			2

#define MAX_EPOCH			1000000

int main()
{
	// �Ű�� ��Ʈ��ũ ����
	int hlayer[2] = { 5,3 };
	MultiLayer.Create(NUM_INPUT, NUM_OUTPUT, 2, hlayer);

	char ch;

	DisplayMenu();

	while ((ch = getche()) != EOF)
	{
		switch (ch)
		{
		case '1':
			Learning_Start();
			break;
		case '2':
			Save_Weight();
			break;
		case '3':
			Load_Weight();
			break;
		case '4':
			Learning_test();
		case '5':
			Test_Net();
			break;
		case '0':
			return 0;	// ����
			break;
		}
		DisplayMenu();
	}

	return 0;
}

void DisplayMenu()
{
	int i;

	char menu[7][64]={	"**********************************",
						"[1]�н�",
						"[2]Weight ����",
						"[3]Weight �б�",
						"[4]�н�������test",
						"[5]test",
						"[0]����"};

	for (i = 0; i < 7; i++)
		printf("\n%s", menu[i]);
	printf("\n���ø޴�:");
}

void Learning_Start()
{
	float learningdata[NUM_TRAINING_SET][NUM_INPUT];
	float dout[NUM_TRAINING_SET][NUM_OUTPUT];

	int i, j;
	int tdata;
	// �н������� ȭ�Ͽ��� �б�
	FILE *fp = fopen("LearningData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>�н������͸� ���� �� �����ϴ�.");
		return;
	}
	for (i = 0; i < NUM_TRAINING_SET; i++)
	{
		for (j = 0; j < NUM_INPUT; j++)
		{
			fscanf(fp, "%d", &tdata);
			learningdata[i][j] = tdata / 255.0;
		}
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			fscanf(fp, "%d", &tdata);
			dout[i][j] = tdata;
		}
	}
	fclose(fp);

	printf("\n=>�н��� �����մϴ�.");
	// �н�
	int epoch,n,p;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// �Է°����� ���� => ������ϱ�,�������н�
		for (n = 0; n < NUM_TRAINING_SET; n++)
		{
			// �Է�����
			for (p = 0; p<NUM_INPUT; p++)
				MultiLayer.pInValue[p] = learningdata[n][p];
			// ���� ����
			for (p = 0; p<NUM_OUTPUT; p++)
				MultiLayer.pCorrectOutValue[p] = dout[n][p];

			// ��°����
			MultiLayer.forward();

			for (p = 0; p<NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p])*(MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

			// �������н� ����ġ����
			MultiLayer.BackPropagationLearning();
		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	printf("\n=>�н�����.");

	
}

void Save_Weight()
{
	printf("\n=>����ġ�� �����մϴ�.");
	MultiLayer.SaveWeight("weight.txt");
}

void Load_Weight()
{
	printf("\n=>����ġ�� �о�ɴϴ�.");
	MultiLayer.LoadWeight("weight.txt");
}

void Learning_test()
{
	printf("\n=>�н������� ��� �˻�\n");

	float learningdata[NUM_TRAINING_SET][NUM_INPUT];
	float dout[NUM_TRAINING_SET][NUM_OUTPUT];

	int i, j;
	int tdata;
	// �н������� ȭ�Ͽ��� �б�
	FILE *fp = fopen("LearningData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>�н������͸� ���� �� �����ϴ�.");
		return;
	}
	for (i = 0; i < NUM_TRAINING_SET; i++)
	{
		for (j = 0; j < NUM_INPUT; j++)
		{
			fscanf(fp, "%d", &tdata);
			learningdata[i][j] = tdata / 255.0;
		}
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			fscanf(fp, "%d", &tdata);
			dout[i][j] = tdata;
		}
	}
	fclose(fp);


	int n, p;
	//������
	for (n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (p = 0; p<NUM_INPUT; p++)
			MultiLayer.pInValue[p] = learningdata[n][p];

		MultiLayer.forward();

		for (p = 0; p < NUM_INPUT; p++)
			printf("%.4f", MultiLayer.pInValue[p]);
		printf("=");
		for (p = 0; p < NUM_OUTPUT; p++)
			printf("%lf,", MultiLayer.pOutValue[p]);
		printf("\n");
	}


}

void Test_Net()
{
	printf("\nTest start\n");
	
	int rgb[3] = { 227,179,165 };

	int p;
	for (p = 0; p<NUM_INPUT; p++)
		MultiLayer.pInValue[p] = rgb[p]/255.0;

	MultiLayer.forward();

	for (p = 0; p < NUM_INPUT; p++)
		printf("%.4f ", MultiLayer.pInValue[p]);
	printf("=");
	for (p = 0; p < NUM_OUTPUT; p++)
		printf("%lf,", MultiLayer.pOutValue[p]);

	if (MultiLayer.pOutValue[0] > 0.8)
		printf("(�Ǻλ��Դϴ�.)\n");
	else
		printf("(�Ǻλ��� �ƴմϴ�.)\n");

}