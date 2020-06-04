#include <stdio.h>

#include "MLP.h"    //자기가 만든건 ""로 추가

CMLP MultiLayer; 


#define NUM_TRAINING_SET 4
#define NUM_INPUT 9
#define NUM_OUTPUT 4

#define MAX_EPOCH 1000000

int main()
{
	int n, p;
	int epoch;

	//신경망 네트워크 구성

	int hlayer[1] = { 2 };
	MultiLayer.Create(NUM_INPUT, NUM_OUTPUT, 1, hlayer);

	//학습데이터 준비

	double x[NUM_TRAINING_SET][NUM_INPUT] = {
						{ 1,1,1
						 ,0,0,1,
						  0,0,1 },  //ㄱ
						 { 1,0,0,
						   1,0,0,
						   1,1,1 }, //ㄴ
						  { 1,1,1
						   ,1,0,0,
							1,1,1 },//ㄷ
							{1,1,1,
							 1,0,1,
							 1,1,1} };  //ㅁ

												//ㄱ정답  //ㄴ정답  //ㄷ정답  //ㅁ정답
	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };   //정답 (목표값)


	if (MultiLayer.LoadWeight("..\\Weight\\weight.txt") == true)
	{
		printf("기존의 가중치로부터 학습을 시작합니다.\n");

	}
	else
	{
		printf("랜덤 가중치로부터 가중치로부터 처음으로 시작합니다.\n");

	}

	//결과출력
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

	// 학습
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		//입력,정답 전달  -> 결과구하기, 역전파학습
		for (n = 0; n < NUM_TRAINING_SET; n++)
		{
			//입력 전달
			for (p = 0;p < NUM_INPUT;p++)
				MultiLayer.pInValue[p] = x[n][p];
			//정답 전달
			for(p=0;p<NUM_OUTPUT;p++)
			MultiLayer.pCorrectOutValue[p] = d[n][p];


			//출력값계산
			MultiLayer.forward();


			for (p = 0;p<NUM_OUTPUT;p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p])*(MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);


			//역전파학습에서 가중치갱신
			MultiLayer.BackPropagationLearning();
		}
		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}

	MultiLayer.SaveWeight("..\\Weight\\weight.txt");

	
	//결과출력
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

