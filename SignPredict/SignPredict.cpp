#include <stdio.h>
#include <conio.h>
#include <math.h>
#define PI 3.1415925521
void DisplayMenu();

void Learning_Start();
void Save_Weight();
void Load_Weight();
void GenTrainingData();
void Test_Net();

#include "MLP.h"
CMLP MultiLayer;

#define NUM_TRAINING_SET	30
#define NUM_INPUT			5
#define NUM_OUTPUT			1

#define MAX_EPOCH			1000000

int main()
{
	// �Ű�� ��Ʈ��ũ ����
	int hlayer[2] = { 10,3 };
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
			GenTrainingData();
			break;
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
						"[4]�н������ͻ���",
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
	float tdata;
	// �н������� ȭ�Ͽ��� �б�
	FILE *fp = fopen("TrainingData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>�н������͸� ���� �� �����ϴ�.");
		return;
	}
	for (i = 0; i < NUM_TRAINING_SET; i++)
	{
		for (j = 0; j < NUM_INPUT; j++)
		{
			fscanf(fp, "%f", &tdata);
			learningdata[i][j] = tdata ;
		}
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			fscanf(fp, "%f", &tdata);
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

void GenTrainingData()
{
	printf("\n=>�н������� ����\n");
	int i,j;
	double value,nextvalue;
	FILE *fp = fopen("TrainingData.txt","wt");

	for(i=0;i<NUM_TRAINING_SET;i++)
	{
		for(j=NUM_INPUT; j>0; j--)
		{
			value = sin(2*PI/(NUM_TRAINING_SET + NUM_INPUT)*(i-j))/2+0.5; //�ϳ��� ���� ���� , i�� ���� ��  // 0~1.0 ������ �� 
			fprintf(fp,"%lf ",value);
		}
		nextvalue = sin(2*PI/(NUM_TRAINING_SET + NUM_INPUT)*(i-j))/2+0.5; //j�� 0�λ��� //���信 �ش�Ǵ� ��
		fprintf(fp,"%lf\n",nextvalue);
	}
	fclose(fp);




}

void Test_Net()
{
	printf("\nTest start\n");

	int i, j;
	double prevalue[NUM_INPUT],value, nextvalue;


	for (j = NUM_INPUT; j > 0; j--)
	{
		value = sin(2 * PI / (NUM_TRAINING_SET + NUM_INPUT)*(4.7 - j)) / 2 + 0.5; //�ϳ��� ���� ���� , i�� ���� ��  // 0~1.0 ������ �� 
		prevalue[NUM_INPUT - j] = value;
	}
	nextvalue = sin(2 * PI / (NUM_TRAINING_SET + NUM_INPUT)*(4.7 - j)) / 2 + 0.5; //j�� 0�λ��� //���信 �ش�Ǵ� ��
	
	int p;
	for (p = 0; p < NUM_INPUT;p++)
		MultiLayer.pInValue[p] = prevalue[p];

	MultiLayer.forward();

	printf("���:%lf,%lf,%lf,%lf,%lf=%lf(%lf)\n",
		MultiLayer.pInValue[0],MultiLayer.pInValue[1],
		MultiLayer.pInValue[2],MultiLayer.pInValue[3],MultiLayer.pInValue[4],
		MultiLayer.pOutValue[0],nextvalue);
	

}
	