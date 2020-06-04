#include <stdio.h>

#include "MLP.h"

CMLP MultiLayer;


int main() 
{
	int p;

	//�Ű�� ��Ʈ��ũ ����

	int hlayer[1] = { 2 };
	MultiLayer.Create(9, 4, 1, hlayer);

	if (MultiLayer.LoadWeight("..\\Weight\\weight.txt") == true)
	{
		printf(" ����ġ�� �о����ϴ�.\n");

	}
	else
	{
		printf("����ġ�� ���� �� �����ϴ�.\n");
		return 0;

	}

	//�׽�Ʈ ������ �Է�
	int test_input[9] = { 1,0,1,
						  1,0,1
						 ,1,1,1 };      //���� ���絵�� ���ϴ°�=>���Ϻм� . �������ڰ� �����.
										//������ �����ʵ��� ������ �н����Ѿ��Ѵ�.  -->> �����׿��� ����

	//�Է�����
	for (p = 0;p < 9;p++)
	{
		MultiLayer.pInValue[p] = test_input[p];
	
	}

	//�����
	MultiLayer.forward();

	//������
	for (p = 0;p < 9;p++)
		printf("%.0f", MultiLayer.pInValue[p]);
	printf("=");

	for (p = 0;p < 4;p++)
		printf("%lf", MultiLayer.pOutValue[p]);
	printf("\n");


	return 0;
}