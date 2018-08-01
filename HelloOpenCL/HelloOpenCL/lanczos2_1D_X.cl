#define cuEPSILON 1e-6
#define PI 3.1415926535897932384626433832795

int matlab_sign(__private int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

float Lanczos2Kernel(float x) {
	float abs_x = fabs((float)x);

	if (-2.0 <= abs_x && abs_x < 2.0) {
		if (x != 0) {
			return (2.0 * sin(PI * abs_x) * sin(PI * abs_x / 2.0)) / (PI * PI * abs_x * abs_x);
		}
		else {
			return 1.0;
		}
	}
	else {
		return 0.0;
	}
}

__kernel
void  interpolation(__global float* data,   //��������
	__global const int* iSmapleNum,//img->height
	__global const int* iLine,//img->width
	__global const int* ResImageW, //����ͼ�����
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
			float res = 0.0;
			// ��ת����õ������ڷ�Χ��
			if (PosXint1 >= 1 && PosXint1 < *iLine - 2 && PosYint1 >= 0 && PosYint1 < *iSmapleNum) {
				for (int i = -1; i <= 2; i++) {
					res += Lanczos2Kernel(PosX - (PosXint1 + i)) * (data)[(PosXint1 + i) * (*iSmapleNum) + PosYint1];
					//res += Lanczos2Kernel(PosY - (PosYint1+i)) * (data)[PosXint1 * (*iSmapleNum) + (PosYint1+i)];
				}
				SCRes[iY*(*ResImageW) + iX] = res > 0 ? res : 0;
			}
			else {
				SCRes[iY*(*ResImageW) + iX] = (data)[PosXint1*(*iSmapleNum) + PosYint1];
			}
		}
	}
}