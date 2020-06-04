#include <stdio.h>

#include "MLP.h"

CMLP MultiLayer;


int main() 
{
	int p;

	//신경망 네트워크 구성

	int hlayer[1] = { 2 };
	MultiLayer.Create(9, 4, 1, hlayer);

	if (MultiLayer.LoadWeight("..\\Weight\\weight.txt") == true)
	{
		printf(" 가중치를 읽었습니다.\n");

	}
	else
	{
		printf("가중치를 읽을 수 없습니다.\n");
		return 0;

	}

	//테스트 데이터 입력
	int test_input[9] = { 1,0,1,
						  1,0,1
						 ,1,1,1 };      //각각 유사도를 구하는것=>패턴분석 . 높은숫자가 가까움.
										//과적합 되지않도록 적당히 학습시켜야한다.  -->> 딥러닝에서 개선

	//입력전달
	for (p = 0;p < 9;p++)
	{
		MultiLayer.pInValue[p] = test_input[p];
	
	}

	//계산결과
	MultiLayer.forward();

	//결과출력
	for (p = 0;p < 9;p++)
		printf("%.0f", MultiLayer.pInValue[p]);
	printf("=");

	for (p = 0;p < 4;p++)
		printf("%lf", MultiLayer.pOutValue[p]);
	printf("\n");


	return 0;
}