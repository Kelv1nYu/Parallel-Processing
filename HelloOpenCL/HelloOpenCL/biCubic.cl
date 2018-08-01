#define cuEPSILON 1e-6

int matlab_sign(int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

float BiCubicPloy(float x) {
	if (x < 0)
		x = -x;
	float a = -0.5;
	if (x <= 1.0)
		return (a + 2) * pow(x, 3) - (a + 3) * pow(x, 2) + 1;
	else if (x <= 2.0)
		return a * pow(x, 3) - 5 * a * pow(x, 2) + 8 * a * x - 4 * a;
	else
		return 0.0;
}

__kernel
void interpolation(__global float* data,   //传入数据
	__global const int* iSmapleNum,//img->height
	__global const int* iLine,//img->width
	__global const int* ResImageW, //传入图像宽度
	__global const int* ResImageH, //图像高度
	__global const int* X,
	__global const int* Y,
	__global const int* TranformHor,
	__global const int* TranformVec,
	__global const float* SectorRadiusPiexl,//扇区半径像素数
	__global const float* ProbeRadiusPixel,
	__global const float* StartAngle,
	__global const float* EndAngle,
	__global const float* AveIntervalAngleReciprocal,//单位角度多少数据
	__global const float* Ratio, //定义的高度/实际高度的像素数=缩放比例
	__global float* SCRes)  //返回结果
{
	int iX = get_global_id(0);
	int iY = get_global_id(1);

	int fHitX = iX - *X + *TranformHor;// 采样位置以交点为原点的成像位置x
	int fHitY = iY - *Y + *TranformVec; // 采样位置以交点为原点的成像位置y
										// Depth
	float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY)); // 距交点距离
																	  // 若映射后的在该圆环内
	if (fHitPointNorm > *ProbeRadiusPixel - cuEPSILON && fHitPointNorm < *SectorRadiusPiexl + cuEPSILON) {
		// 映射后的点的s角度
		float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos求角度
																									// 若映射后的点的角度没有超出范围
		if (fSamplePointAngle > *StartAngle - cuEPSILON && fSamplePointAngle < *EndAngle + cuEPSILON) {
			// CublicSample
			float PosX = (fSamplePointAngle - *StartAngle)*(*AveIntervalAngleReciprocal); // 第几条数据
			float PosY = (fHitPointNorm - *ProbeRadiusPixel)*(*Ratio);  // 第几条数据的第几个数据
			int PosXint1 = floor(PosX);
			int PosYint1 = floor(PosY);
			float result = 0;
			float dist_x, dist_y;
			// 若转换后该点区间在范围内
			if (PosXint1 >= 1 && PosXint1 < *iLine - 2 && PosYint1 >= 1 && PosYint1 < *iSmapleNum - 2) {
				//取出映射到原图中的点的四周的16个点的坐标 
				for (int i = -1; i < 3; i++) {
					for (int j = -1; j < 3; j++) {
						dist_x = BiCubicPloy(PosX - (PosXint1 + i));
						dist_y = BiCubicPloy(PosY - (PosYint1 + j));
						result += dist_x * dist_y * (data)[(PosXint1 + i)*(*iSmapleNum) + PosYint1 + j];
					}
				}
				SCRes[iY*(*ResImageW) + iX] = result > 0 ? result : 0;
			}
			else {
				SCRes[iY*(*ResImageW) + iX] = (data)[PosXint1*(*iSmapleNum) + PosYint1];
			}
		}
	}
}
