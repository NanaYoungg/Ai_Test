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
	// 신경망 네트워크 구성
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
						"[4]학습데이터test",
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
	int tdata;
	// 학습데이터 화일에서 읽기
	FILE *fp = fopen("LearningData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>학습데이터를 읽을 수 없습니다.");
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

void Learning_test()
{
	printf("\n=>학습데이터 결과 검사\n");

	float learningdata[NUM_TRAINING_SET][NUM_INPUT];
	float dout[NUM_TRAINING_SET][NUM_OUTPUT];

	int i, j;
	int tdata;
	// 학습데이터 화일에서 읽기
	FILE *fp = fopen("LearningData.txt", "rt");
	if (fp == NULL)
	{
		printf("\n=>학습데이터를 읽을 수 없습니다.");
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
	//결과출력
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
		printf("(피부색입니다.)\n");
	else
		printf("(피부색이 아닙니다.)\n");

}