#define cuEPSILON 1e-6

int matlab_sign(__private int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

float BiCubicPloy(float x) {
	float abs_x = fabs((float)x);//ȡx�ľ���ֵ
	float a = -0.5;
	if (abs_x <= 1.0)
		return (a + 2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
	else if (abs_x <= 2.0)
		return a * pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
	else
		return 0.0;
}

__kernel
void interpolation(__global float* data,   //��������
	__global const int* iSmapleNum,//img->height
	__global const int* iLine,//img->width
	__global const int* ResImageW, //����ͼ����
	__global const int* ResImageH, //ͼ��߶�
	__global const int* X,
	__global const int* Y,
	__global const int* TranformHor,
	__global const int* TranformVec,
	__global const float* SectorRadiusPiexl,//�����뾶������
	__global const float* ProbeRadiusPixel,
	__global const float* StartAngle,
	__global const float* EndAngle,
	__global const float* AveIntervalAngleReciprocal,//��λ�Ƕȶ�������
	__global const float* Ratio, //����ĸ߶�/ʵ�ʸ߶ȵ�������=���ű���
	__global float* SCRes)  //���ؽ��
{
	int iX = get_global_id(0);
	int iY = get_global_id(1);

	int fHitX = iX - *X + *TranformHor;// ����λ���Խ���Ϊԭ��ĳ���λ��x
	int fHitY = iY - *Y + *TranformVec; // ����λ���Խ���Ϊԭ��ĳ���λ��y
										// Depth
	float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY)); // �ཻ�����
																	  // ��ӳ�����ڸ�Բ����
	if (fHitPointNorm > *ProbeRadiusPixel - cuEPSILON && fHitPointNorm < *SectorRadiusPiexl + cuEPSILON) {
		// ӳ���ĵ�ĽǶ�
		float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos��Ƕ�
																									// ��ӳ���ĵ�ĽǶ�û�г�����Χ
		if (fSamplePointAngle > *StartAngle - cuEPSILON && fSamplePointAngle < *EndAngle + cuEPSILON) {
			// CublicSample
			float PosX = (fSamplePointAngle - *StartAngle)*(*AveIntervalAngleReciprocal); // �ڼ�������
			float PosY = (fHitPointNorm - *ProbeRadiusPixel)*(*Ratio);  // �ڼ������ݵĵڼ�������
			int PosXint1 = floor(PosX);
			int PosYint1 = floor(PosY);
			float u = PosX - PosXint1;
			float v = PosY - PosYint1;
			float res = 0.0;
			// ��ת����õ������ڷ�Χ��
			if (PosXint1 >= 0 && PosXint1 < *iLine - 1 && PosYint1 >= 0 && PosYint1 < *iSmapleNum - (*ResImageH / 2)) {
				SCRes[iY*(*ResImageW) + iX] = (1 - u)*(1 - v)*(data)[PosXint1*(*iSmapleNum) + PosYint1] +
					(1 - u)*v*(data)[PosXint1*(*iSmapleNum) + (PosYint1 + 1)] +
					u * (1 - v)*(data)[(PosXint1 + 1)*(*iSmapleNum) + PosYint1] +
					u * v*(data)[(PosXint1 + 1)*(*iSmapleNum) + (PosYint1 + 1)];
			}
			else if (PosXint1 >= 1 && PosXint1 < *iLine - 2 && PosYint1 >= *iSmapleNum - (*ResImageH / 2) && PosYint1 < *iSmapleNum - 2) {
				for (int i = -1; i < 3; i++) {
					for (int j = -1; j < 3; j++) {
						res += BiCubicPloy(PosX - (PosXint1 + i))*BiCubicPloy(PosY - (PosYint1 + j)) *
							(data)[(PosXint1 + i)*(*iSmapleNum) + PosYint1 + j];
					}
				}
				SCRes[iY*(*ResImageW) + iX] = res > 0 ? res : 0;
			}
			if (PosXint1 == *iLine - 1 || PosYint1 == *iSmapleNum - 1)
				SCRes[iY*(*ResImageW) + iX] = (data)[PosXint1*(*iSmapleNum) + PosYint1];
		}
	}
}
