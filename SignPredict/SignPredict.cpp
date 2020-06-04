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
	// 신경망 네트워크 구성
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
			return 0;	// 종료
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
						"[1]학습",
						"[2]Weight 저장",
						"[3]Weight 읽기",
						"[4]학습데이터생성",
						"[5]test",
						"[0]종료"};

	for (i = 0; i < 7; i++)
		printf("\n%s", menu[i]);
	printf("\n선택메뉴:");
}

void Learning_Start()
{
	float learningdata[NUM_TRAINING_SET][NUM_INPUT];
	float dout[NUM_TRAINING_SET][NUM_OUTPUT];

	int i, j;
	float tdata;
	// 학습데이터 화일에서 읽기
	FILE *fp = fopen("TrainingData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>학습데이터를 읽을 수 없습니다.");
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

	printf("\n=>학습을 시작합니다.");
	// 학습
	int epoch,n,p;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// 입력과정답 전달 => 결과구하기,역전파학습
		for (n = 0; n < NUM_TRAINING_SET; n++)
		{
			// 입력전달
			for (p = 0; p<NUM_INPUT; p++)
				MultiLayer.pInValue[p] = learningdata[n][p];
			// 정답 전달
			for (p = 0; p<NUM_OUTPUT; p++)
				MultiLayer.pCorrectOutValue[p] = dout[n][p];

			// 출력값계산
			MultiLayer.forward();

			for (p = 0; p<NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p])*(MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

			// 역전파학습 가중치갱신
			MultiLayer.BackPropagationLearning();
		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	printf("\n=>학습종료.");

	
}

void Save_Weight()
{
	printf("\n=>가중치를 저장합니다.");
	MultiLayer.SaveWeight("weight.txt");
}

void Load_Weight()
{
	printf("\n=>가중치를 읽어옵니다.");
	MultiLayer.LoadWeight("weight.txt");
}

void GenTrainingData()
{
	printf("\n=>학습데이터 생성\n");
	int i,j;
	double value,nextvalue;
	FILE *fp = fopen("TrainingData.txt","wt");

	for(i=0;i<NUM_TRAINING_SET;i++)
	{
		for(j=NUM_INPUT; j>0; j--)
		{
			value = sin(2*PI/(NUM_TRAINING_SET + NUM_INPUT)*(i-j))/2+0.5; //하나의 각도 간격 , i는 시작 값  // 0~1.0 사이의 값 
			fprintf(fp,"%lf ",value);
		}
		nextvalue = sin(2*PI/(NUM_TRAINING_SET + NUM_INPUT)*(i-j))/2+0.5; //j는 0인상태 //정답에 해당되는 값
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
		value = sin(2 * PI / (NUM_TRAINING_SET + NUM_INPUT)*(4.7 - j)) / 2 + 0.5; //하나의 각도 간격 , i는 시작 값  // 0~1.0 사이의 값 
		prevalue[NUM_INPUT - j] = value;
	}
	nextvalue = sin(2 * PI / (NUM_TRAINING_SET + NUM_INPUT)*(4.7 - j)) / 2 + 0.5; //j는 0인상태 //정답에 해당되는 값
	
	int p;
	for (p = 0; p < NUM_INPUT;p++)
		MultiLayer.pInValue[p] = prevalue[p];

	MultiLayer.forward();

	printf("결과:%lf,%lf,%lf,%lf,%lf=%lf(%lf)\n",
		MultiLayer.pInValue[0],MultiLayer.pInValue[1],
		MultiLayer.pInValue[2],MultiLayer.pInValue[3],MultiLayer.pInValue[4],
		MultiLayer.pOutValue[0],nextvalue);
	

}
	