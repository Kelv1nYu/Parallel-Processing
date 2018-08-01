#define cuEPSILON 1e-6

int matlab_sign(__private int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

__kernel
void interpolation(__global float* data,   //��������
	__global const int* iSmapleNum,//img->height
	__global const int* iLine,//img->width
	__global const int* ResImageW, //����ͼ����
	__global const int* ResImageH, //ͼ��߶�
	__global const int* SampleOriginX,
	__global const int* SampleOriginY,
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

	int fHitX = iX - *SampleOriginX + *TranformHor;// ����λ���Խ���Ϊԭ��ĳ���λ��x
	int fHitY = iY - *SampleOriginY + *TranformVec; // ����λ���Խ���Ϊԭ��ĳ���λ��y
	// Depth
	float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY)); // �ཻ�����
																	  // ��ӳ�����ڸ�Բ����
	if (fHitPointNorm > *ProbeRadiusPixel - cuEPSILON && fHitPointNorm < *SectorRadiusPiexl + cuEPSILON) {
		float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos��Ƕ�
																									// ��ӳ���ĵ�ĽǶ�û�г�����Χ
		if (fSamplePointAngle > *StartAngle - cuEPSILON && fSamplePointAngle < *EndAngle + cuEPSILON) {
			// CublicSample
			float PosX = (fSamplePointAngle - *StartAngle) * (*AveIntervalAngleReciprocal); // �ڼ�������
			float PosY = (fHitPointNorm - *ProbeRadiusPixel) * (*Ratio);  // �ڼ������ݵĵڼ�������
			int PosXint1 = floor(PosX);
			int PosYint1 = floor(PosY);
			float u = PosX - PosXint1;
			float v = PosY - PosYint1;
			if (u > 0.5)
				PosXint1 = ceil(PosX);
			if (v > 0.5)
				PosYint1 = ceil(PosY);

			// ��ת����õ������ڷ�Χ��
			if (PosXint1 >= 0 && PosXint1<*iLine && PosYint1 >= 0 && PosYint1<*iSmapleNum)
				SCRes[iY* (*ResImageW) + iX] = (data)[PosXint1* (*iSmapleNum) + PosYint1];
		}
	}

}
