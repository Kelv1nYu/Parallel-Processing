#define cuEPSILON 1e-6

int matlab_sign(__private int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

__kernel
void interpolation(__global float* data,   //传入数据
	__global const int* iSmapleNum,//img->height
	__global const int* iLine,//img->width
	__global const int* ResImageW, //传入图像宽度
	__global const int* ResImageH, //图像高度
	__global const int* SampleOriginX,
	__global const int* SampleOriginY,
	__global const int* TranformHor,
	__global const int* TranformVec,
	__global const float* SectorRadiusPixel,//扇区半径像素数
	__global const float* ProbeRadiusPixel,
	__global const float* StartAngle,
	__global const float* EndAngle,
	__global const float* AveIntervalAngleReciprocal,//单位角度多少数据
	__global const float* Ratio, //定义的高度/实际高度的像素数=缩放比例
	__global float* SCRes)  //返回结果
{
	int iX = get_global_id(0);
	int iY = get_global_id(1);

	int fHitX = iX - *SampleOriginX + *TranformHor;// 采样位置以交点为原点的成像位置x
	int fHitY = iY - *SampleOriginY + *TranformVec; // 采样位置以交点为原点的成像位置y
	// Depth
	float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY));

	if (fHitPointNorm > *ProbeRadiusPixel - cuEPSILON && fHitPointNorm < *SectorRadiusPixel + cuEPSILON) 
	{
		// Angle
		float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos求角度
																									// 若映射后的点的角度没有超出范围
		if (fSamplePointAngle > *StartAngle - cuEPSILON && fSamplePointAngle < *EndAngle + cuEPSILON) 
		{
			// CublicSample
			float PosX = (fSamplePointAngle - *StartAngle)*(*AveIntervalAngleReciprocal); // 第几条数据
			float PosY = (fHitPointNorm - *ProbeRadiusPixel)*(*Ratio);  // 第几条数据的第几个数据
			int PosXint1 = floor(PosX);
			int PosYint1 = floor(PosY);
			float u = PosX - PosXint1;
			float v = PosY - PosYint1;
			// 控制边界，防止越界
			if (PosXint1 >= 0 && PosXint1 < *iLine - 1 && PosYint1 >= 0 && PosYint1 < *iSmapleNum - 1) 
			{
				SCRes[iY * (*ResImageW) + iX]	=	(1 - u) * (1 - v) * (data)[PosXint1 * (*iSmapleNum) + PosYint1] 
												+	(1 - u) *    v	  * (data)[PosXint1 * (*iSmapleNum) + (PosYint1 + 1)] 
												+	   u    * (1 - v) * (data)[(PosXint1 + 1) * (*iSmapleNum) + PosYint1] 
												+	   u    *    v    * (data)[(PosXint1 + 1) * (*iSmapleNum) + (PosYint1 + 1)];
			}
			if (PosXint1 == *iLine - 1 || PosYint1 == *iSmapleNum - 1)
				SCRes[iY * (*ResImageW) + iX] = (data)[PosXint1*(*iSmapleNum) + PosYint1];
		}
	}
}
