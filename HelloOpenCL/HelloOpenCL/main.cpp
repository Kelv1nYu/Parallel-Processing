// OpenCl-1.cpp : Defines the entry point for the console application.
//
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BINSIZE 256
#define PI 3.1415926535897932384626433832795
#define cuEPSILON 1e-6

bool convertCLFile2String(const char *pFileName, std::string& Str);

cl_context CreateContext();

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);

void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel);


// 图片结构体
struct Image {
	float *data;
	int width, height;
	float depth, angle, radius;//扫描深度， Curve探头扫描夹角， Curve 探头半径
	Image() :width(0), height(0), depth(0.0), angle(0.0), radius(0.0) {}
};


// 把float数组转换成opencv图片对象
IplImage* floatArrayToOpencvImage(float* data, int width, int height) {
	IplImage* img = cvCreateImage(CvSize(width, height), 8, 1);

	for (int j = 0; j<height; j++)
	{
		for (int i = 0; i<width; i++)
			img->imageData[j*width + i] = data[j*width + i];
	}
	return img;
}

int matlab_sign(int x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

//计算双三次插值的权重
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
//兰索斯权值
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

IplImage* convert(Image *img) {
	int iSmapleNum = img->height;
	int iLine = img->width;
	int Lnum = iLine;
	int Snum = iSmapleNum;

	int iBHeightResolve = 512; // 预定义的图像高度							
	float ProbeRadiusPixel = img->radius * Snum / img->depth; 
	float SectorRadiusPixel = ProbeRadiusPixel + Snum;  // 扇区半径的像素数
	float StartAngle = -(img->angle);
	float EndAngle = img->angle;
	float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // 单位角度有多少条数据
	int ResImageH = iBHeightResolve;
	float Ratio = ResImageH / (SectorRadiusPixel - cos(img->angle)*ProbeRadiusPixel + 1); // 定义的高度/实际高度的像素数=缩放比例
	int ResImageW = round(sin(img->angle)*SectorRadiusPixel * 2 * Ratio + 1); // 图像宽度
	ProbeRadiusPixel = ProbeRadiusPixel * Ratio;    // 按比例缩放后的探测半径
	SectorRadiusPixel = SectorRadiusPixel * Ratio; // 按比例缩放后的扇区半径
	Ratio = 1 / Ratio;   // 原高度/定义高度

	//坐标转化参数
	int SampleOriginX = ResImageW / 2;
	int SampleOriginY = 0;
	int TranformHor = 0;
	int TranformVec = SectorRadiusPixel - ResImageH; // 扇区半径-图像高度

	float *SCRes = new float[ResImageH * ResImageW];


	cl_context context = 0;//OpenCL上下文
	cl_command_queue commandQueue = 0;//命令队列
	cl_program program = 0;//程序对象
	cl_device_id device = 0;//设备信息
	cl_kernel kernel = 0;//内核对象
	cl_int errNum = 0;//状态信息，没有错误则状态为0
	cl_mem memObjects[16];//内存对象
	cl_event event;
	cl_ulong start_time, end_time;
	double total_time;

	// 一、选择OpenCL平台并创建一个上下文
	context = CreateContext();

	// 二、 创建设备并创建命令队列
	commandQueue = CreateCommandQueue(context, &device);

	cout << "1.nearest" << endl;
	cout << "2.bilinear" << endl;
	cout << "3.biCubic" << endl;
	cout << "4.lanczos" << endl;
	cout << "5.Ver_Bilinear_Cubic" << endl;
	cout << "6.Ver_Lanczos2_2D_Cubic" << endl;
	int input;
	cin >> input;
	switch (input)
	{
	//创建和构建程序对象
	case 1:
		program = CreateProgram(context, device, "nearest.cl");
		break;
	case 2:
		program = CreateProgram(context, device, "bilinear.cl");
		break;
	case 3:
		program = CreateProgram(context, device, "biCubic.cl");
		break;
	case 4:
		program = CreateProgram(context, device, "Lanczos2_2D.cl");
		break;
	case 5:
		program = CreateProgram(context, device, "Ver_Bilinear_BiCubic.cl");
		break;
	case 6:
		program = CreateProgram(context, device, "Ver_Lanczos2_2D_BiCubic.cl");
		break;
	default:
		cout << "无效输入" << endl;
		cout << "默认使用nearest.cl" << endl;
		program = CreateProgram(context, device, "nearest.cl");
		break;
	}
	

	// 四、 创建OpenCL内核并分配内存空间
	kernel = clCreateKernel(program, "interpolation", NULL);

	//创建内存对象
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * iSmapleNum * iLine, img->data, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &iSmapleNum, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &iLine, NULL);
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &ResImageW, NULL);
	memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &ResImageH, NULL);
	memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &SampleOriginX, NULL);
	memObjects[6] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &SampleOriginY, NULL);
	memObjects[7] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &TranformHor, NULL);
	memObjects[8] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int), &TranformVec, NULL);
	memObjects[9] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &SectorRadiusPixel, NULL);
	memObjects[10] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &ProbeRadiusPixel, NULL);
	memObjects[11] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &StartAngle, NULL);
	memObjects[12] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &EndAngle, NULL);
	memObjects[13] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &AveIntervalAngleReciprocal, NULL);
	memObjects[14] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), &Ratio, NULL);
	memObjects[15] = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float) * ResImageH * ResImageW, NULL, NULL);

	// 五、 设置内核数据并执行内核
	for (int i = 0; i < 16; i++) {
		errNum |= clSetKernelArg(kernel, i, sizeof(cl_mem), &memObjects[i]);
	}
	if (errNum != CL_SUCCESS) {
		cout << "创建内存对象失败！" << endl;
	}

	//计算所需工作组
	size_t globalWorkSize[2];
	globalWorkSize[0] = ResImageW;
	globalWorkSize[1] = ResImageH;
	size_t localWorkSize[2];
	localWorkSize[0] = 1;
	localWorkSize[1] = 1;

	
	//利用命令队列对要在设备上执行的内核排队
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, &event);

	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "执行队列失败！" << endl;
	}

	clFinish(commandQueue);
	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	if (errNum != CL_SUCCESS) {
		cout << "clGetEventProfilingInfo失败！" << endl;
	}

	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	if (errNum != CL_SUCCESS) {
		cout << "clGetEventProfilingInfo失败！" << endl;
	}

	// 六、 读取执行结果
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[15], CL_TRUE,
		0, ResImageH * ResImageW * sizeof(float), SCRes,
		0, NULL, NULL);

	if (errNum != CL_SUCCESS) {
		cout << "读取结果失败！" << endl;
	}

	total_time = end_time - start_time;
	printf("\nGPU run time = %0.3f ms\n", (total_time / 1000000.0));

	//释放OpenCL资源
	for (int i = 0; i < 16; i++) {
		clReleaseMemObject(memObjects[i]);
	}
	Cleanup(context, commandQueue, program, kernel);

	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	//使用CPU计算
	cpu_start = clock();
	for (int iX = 0; iX < ResImageW; iX++) {
		for (int iY = 0; iY < ResImageH; iY++) {
			int fHitX = iX - SampleOriginX + TranformHor;// 采样位置以交点为原点的成像位置x
			int fHitY = iY - SampleOriginY + TranformVec; // 采样位置以交点为原点的成像位置y
			float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY)); // 距交点距离												  // 若映射后的在该圆环内
			if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON && fHitPointNorm < SectorRadiusPixel + cuEPSILON) {
				float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos求角度
																											// 若映射后的点的角度没有超出范围
				if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
					// CublicSample
					float PosX = (fSamplePointAngle - StartAngle) * (AveIntervalAngleReciprocal); // 第几条数据
					float PosY = (fHitPointNorm - ProbeRadiusPixel) * (Ratio);  // 第几条数据的第几个数据
					int PosXint1 = floor(PosX);
					int PosYint1 = floor(PosY);
					float u = PosX - PosXint1;
					float v = PosY - PosYint1;
					float result = 0;
					switch (input)
					{	
					case 1:
						if (u > 0.5)
							PosXint1 = ceil(PosX);
						if (v > 0.5)
							PosYint1 = ceil(PosY);
						// 若转换后该点区间在范围内
						if (PosXint1 >= 0 && PosXint1<iLine && PosYint1 >= 0 && PosYint1<iSmapleNum)
							SCRes[iY* (ResImageW) + iX] = (img->data)[PosXint1* (iSmapleNum) + PosYint1];
						break;
					case 2:
						if (PosXint1 >= 0 && PosXint1 < iLine - 1 && PosYint1 >= 0 && PosYint1 < iSmapleNum - 1)
						{
							SCRes[iY * (ResImageW) + iX] = (1 - u) * (1 - v) * (img->data)[PosXint1 * (iSmapleNum) + PosYint1]
								+ (1 - u) *    v	  * (img->data)[PosXint1 * (iSmapleNum) + (PosYint1 + 1)]
								+ u * (1 - v) * (img->data)[(PosXint1 + 1) * (iSmapleNum) + PosYint1]
								+ u * v    * (img->data)[(PosXint1 + 1) * (iSmapleNum) + (PosYint1 + 1)];
						}
						if (PosXint1 == iLine - 1 || PosYint1 == iSmapleNum - 1)
							SCRes[iY * (ResImageW) + iX] = (img->data)[PosXint1*(iSmapleNum) + PosYint1];
						
						break;
					case 3:
						float dist_x, dist_y;
						// 若转换后该点区间在范围内
						if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= 1 && PosYint1 < iSmapleNum - 2) {
							//取出映射到原图中的点的四周的16个点的坐标 
							for (int i = -1; i < 3; i++) {
								for (int j = -1; j < 3; j++) {
									dist_x = BiCubicPloy(PosX - (PosXint1 + i));
									dist_y = BiCubicPloy(PosY - (PosYint1 + j));
									result += dist_x * dist_y * (img->data)[(PosXint1 + i)*(iSmapleNum) + PosYint1 + j];
								}
							}
							SCRes[iY*(ResImageW) + iX] = result > 0 ? result : 0;
						}
						else {
							SCRes[iY*(ResImageW) + iX] = (img->data)[PosXint1*(iSmapleNum) + PosYint1];
						}
						break;
					case 4:
						// 若转换后该点区间在范围内
						if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= 1 && PosYint1 < iSmapleNum - 2) {
							for (int i = -1; i <= 2; i++) {
								for (int j = -1; j <= 2; j++) {
									result += Lanczos2Kernel(PosX - (PosXint1 + i)) * Lanczos2Kernel(PosY - (PosYint1 + j)) 
										* (img->data)[(PosXint1 + i) * (iSmapleNum) + PosYint1 + j];
								}
							}
							SCRes[iY*(ResImageW) + iX] = result > 0 ? result : 0;
						}
						else {
							SCRes[iY*(ResImageW) + iX] = (img->data)[PosXint1*(iSmapleNum) + PosYint1];
						}
						break;
					case 5:
						if (PosXint1 >= 0 && PosXint1 < iLine - 1 && PosYint1 >= 0 && PosYint1 < iSmapleNum - (ResImageH / 2)) {
							SCRes[iY*(ResImageW) + iX] = (1 - u)*(1 - v)*(img->data)[PosXint1*(iSmapleNum) + PosYint1] +
								(1 - u)*v*(img->data)[PosXint1*(iSmapleNum) + (PosYint1 + 1)] +
								u * (1 - v)*(img->data)[(PosXint1 + 1)*(iSmapleNum) + PosYint1] +
								u * v*(img->data)[(PosXint1 + 1)*(iSmapleNum) + (PosYint1 + 1)];
						}
						else if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= iSmapleNum - (ResImageH / 2) && PosYint1 < iSmapleNum - 2) {
							for (int i = -1; i < 3; i++) {
								for (int j = -1; j < 3; j++) {
									result += BiCubicPloy(PosX - (PosXint1 + i))*BiCubicPloy(PosY - (PosYint1 + j)) *
										(img->data)[(PosXint1 + i)*(iSmapleNum) + PosYint1 + j];
								}
							}
							SCRes[iY*(ResImageW) + iX] = result > 0 ? result : 0;
						}
						if (PosXint1 == iLine - 1 || PosYint1 == iSmapleNum - 1)
							SCRes[iY*(ResImageW) + iX] = (img->data)[PosXint1*(iSmapleNum) + PosYint1];
						break;
					case 6:
						// 若转换后该点区间在范围内
						if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= 1 && PosYint1 < iSmapleNum - (ResImageH / 2)) {
							for (int i = -1; i <= 2; i++) {
								for (int j = -1; j <= 2; j++) {
									result += Lanczos2Kernel(PosX - (PosXint1 + i)) * Lanczos2Kernel(PosY - (PosYint1 + j)) * (img->data)[(PosXint1 + i) * (iSmapleNum) + PosYint1 + j];
								}
							}
							SCRes[iY*(ResImageW) + iX] = result > 0 ? result : 0;
						}
						else if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= iSmapleNum - (ResImageH / 2) && PosYint1 < iSmapleNum - 2) {
							for (int i = -1; i < 3; i++) {
								for (int j = -1; j < 3; j++) {
									result += BiCubicPloy(PosX - (PosXint1 + i))*BiCubicPloy(PosY - (PosYint1 + j)) *(img->data)[(PosXint1 + i)*(iSmapleNum) + (PosYint1 + j)];
								}
							}
							SCRes[iY*(ResImageW) + iX] = result > 0 ? result : 0;
						}
						else {
							SCRes[iY*(ResImageW) + iX] = (img->data)[PosXint1*(iSmapleNum) + PosYint1];
						}
						break;
					default:
						cout << "无效输入" << endl;
						break;
					}
	
				}
			}
		}

	}

	cpu_end = clock();
	cout << "CPU run time = "<<(float)(cpu_end - cpu_start) * 1000  / CLOCKS_PER_SEC << " ms" << endl;

	return floatArrayToOpencvImage(SCRes, ResImageW, ResImageH);
}


//主函数
int main(int argc, char* argv[])
{
	Image *img = new Image();
	FILE *file;
	if (!(file = fopen("C:/Users/Administrator/source/repos/HelloOpenCL/HelloOpenCL/image.dat", "rb")))
	{
		perror("fopen()");
		return NULL;
	}

	int status = fscanf(file, "%d %d %f %f %f\n",
		&(img->width), &(img->height),
		&(img->depth), &(img->angle),
		&(img->radius));
	if (5 != status) {
		printf("读取图片信息出错!");
		return 0;
	}

	
	// 读取图片并转换成opencv图片对象
	int cnt = img->width * img->height;
	img->data = (float*)malloc(cnt * sizeof(float));
	fread(img->data, sizeof(float), cnt, file);


	IplImage* im = floatArrayToOpencvImage(img->data,img->width,img->height);
	IplImage* imsc = convert(img);
	Mat im_mat = cvarrToMat(im);
	Mat imsc_mat = cvarrToMat(imsc);

	cvShowImage("原图片", im);
	cvShowImage("转换后图片", imsc);

	imwrite("C:/Users/Administrator/Desktop/res_image.jpg", imsc_mat);
	cvWaitKey();
	cvReleaseImage(&im);
	cvReleaseImage(&imsc);

	return 0;


}

// 将cl文件代码转为字符串
bool convertCLFile2String(const char *pFileName, std::string& Str)
{
	size_t        uiSize = 0;
	size_t        uiFileSize = 0;
	char        *pStr = NULL;
	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));
	if (fFile.is_open())
	{
		fFile.seekg(0, std::fstream::end);
		uiSize = uiFileSize = (size_t)fFile.tellg();  // 获得文件大小
		fFile.seekg(0, std::fstream::beg);
		pStr = new char[uiSize + 1];
		if (NULL == pStr)
		{
			fFile.close();
			return 0;
		}
		fFile.read(pStr, uiFileSize);                // 读取uiFileSize字节
		fFile.close();
		pStr[uiSize] = '\0';
		Str = pStr;
		delete[] pStr;
		return true;
	}

	cerr << "Error: Failed to open cl file\n:" << pFileName << endl;

	return 0;
}

//选择OpenCL平台并创建一个上下文
cl_context CreateContext()
{
	cl_int errNum;//状态
	cl_uint numPlatforms;//所有运行平台
	cl_platform_id firstPlatformId;//第一个运行平台
	cl_context context = NULL;//OpenCL上下文
 //选择可用的平台中的第一个
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);//获取平台ID
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	//用第一个平台创建一个OpenCL上下文环境
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};//构件上下文属性结构体
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);//创建上下文
	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "创建上下文失败" << endl;
		exit(-1);
	}

	return context;
}

//创建设备并创建命令队列
//根据传入的上下文进行
//通过指针返回设备Id
//返回命令队列
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;//状态
	cl_device_id *devices;//所有设备
	cl_command_queue commandQueue = NULL;//命令队列
	size_t deviceBufferSize = -1;//设备缓存大小

								 // 获取设备缓冲区大小
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// 为设备分配缓存空间
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);//获取设备上下文信息
	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "创建上下文信息失败" << endl;
	}

	//选取可用设备中的第一个，创建命令队列
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	*device = devices[0];//通过指针传出第一个设备信息
	delete[] devices;//释放
	return commandQueue;//返回命令队列
}

//创建和构建程序对象
//传入OpenCL的上下文信息、设备信息和核函数文件
//从核函数文件中加载核函数代码，编译到创建的程序对象中
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;//状态
	cl_program program;//程序对象

					   //读取核函数源代码
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();//将源代码从string类型转为C字符串类型。
										   //用上下文和源代码创建程序对象
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);

	//编译程序对象
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "编译程序对象失败" << endl;
	}

	//返回程序对象
	return program;
}

// 释放OpenCL资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel)
{
	//释放命令队列
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	//释放内核对象
	if (kernel != 0)
		clReleaseKernel(kernel);

	//释放程序对象
	if (program != 0)
		clReleaseProgram(program);

	//释放上下文
	if (context != 0)
		clReleaseContext(context);
}


