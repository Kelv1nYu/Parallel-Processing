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


// ͼƬ�ṹ��
struct Image {
	float *data;
	int width, height;
	float depth, angle, radius;//ɨ����ȣ� Curve̽ͷɨ��нǣ� Curve ̽ͷ�뾶
	Image() :width(0), height(0), depth(0.0), angle(0.0), radius(0.0) {}
};


// ��float����ת����opencvͼƬ����
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

//����˫���β�ֵ��Ȩ��
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
//����˹Ȩֵ
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

	int iBHeightResolve = 512; // Ԥ�����ͼ��߶�							
	float ProbeRadiusPixel = img->radius * Snum / img->depth; 
	float SectorRadiusPixel = ProbeRadiusPixel + Snum;  // �����뾶��������
	float StartAngle = -(img->angle);
	float EndAngle = img->angle;
	float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // ��λ�Ƕ��ж���������
	int ResImageH = iBHeightResolve;
	float Ratio = ResImageH / (SectorRadiusPixel - cos(img->angle)*ProbeRadiusPixel + 1); // ����ĸ߶�/ʵ�ʸ߶ȵ�������=���ű���
	int ResImageW = round(sin(img->angle)*SectorRadiusPixel * 2 * Ratio + 1); // ͼ����
	ProbeRadiusPixel = ProbeRadiusPixel * Ratio;    // ���������ź��̽��뾶
	SectorRadiusPixel = SectorRadiusPixel * Ratio; // ���������ź�������뾶
	Ratio = 1 / Ratio;   // ԭ�߶�/����߶�

	//����ת������
	int SampleOriginX = ResImageW / 2;
	int SampleOriginY = 0;
	int TranformHor = 0;
	int TranformVec = SectorRadiusPixel - ResImageH; // �����뾶-ͼ��߶�

	float *SCRes = new float[ResImageH * ResImageW];


	cl_context context = 0;//OpenCL������
	cl_command_queue commandQueue = 0;//�������
	cl_program program = 0;//�������
	cl_device_id device = 0;//�豸��Ϣ
	cl_kernel kernel = 0;//�ں˶���
	cl_int errNum = 0;//״̬��Ϣ��û�д�����״̬Ϊ0
	cl_mem memObjects[16];//�ڴ����
	cl_event event;
	cl_ulong start_time, end_time;
	double total_time;

	// һ��ѡ��OpenCLƽ̨������һ��������
	context = CreateContext();

	// ���� �����豸�������������
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
	//�����͹����������
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
		cout << "��Ч����" << endl;
		cout << "Ĭ��ʹ��nearest.cl" << endl;
		program = CreateProgram(context, device, "nearest.cl");
		break;
	}
	

	// �ġ� ����OpenCL�ں˲������ڴ�ռ�
	kernel = clCreateKernel(program, "interpolation", NULL);

	//�����ڴ����
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

	// �塢 �����ں����ݲ�ִ���ں�
	for (int i = 0; i < 16; i++) {
		errNum |= clSetKernelArg(kernel, i, sizeof(cl_mem), &memObjects[i]);
	}
	if (errNum != CL_SUCCESS) {
		cout << "�����ڴ����ʧ�ܣ�" << endl;
	}

	//�������蹤����
	size_t globalWorkSize[2];
	globalWorkSize[0] = ResImageW;
	globalWorkSize[1] = ResImageH;
	size_t localWorkSize[2];
	localWorkSize[0] = 1;
	localWorkSize[1] = 1;

	
	//����������ж�Ҫ���豸��ִ�е��ں��Ŷ�
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, &event);

	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "ִ�ж���ʧ�ܣ�" << endl;
	}

	clFinish(commandQueue);
	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	if (errNum != CL_SUCCESS) {
		cout << "clGetEventProfilingInfoʧ�ܣ�" << endl;
	}

	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	if (errNum != CL_SUCCESS) {
		cout << "clGetEventProfilingInfoʧ�ܣ�" << endl;
	}

	// ���� ��ȡִ�н��
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[15], CL_TRUE,
		0, ResImageH * ResImageW * sizeof(float), SCRes,
		0, NULL, NULL);

	if (errNum != CL_SUCCESS) {
		cout << "��ȡ���ʧ�ܣ�" << endl;
	}

	total_time = end_time - start_time;
	printf("\nGPU run time = %0.3f ms\n", (total_time / 1000000.0));

	//�ͷ�OpenCL��Դ
	for (int i = 0; i < 16; i++) {
		clReleaseMemObject(memObjects[i]);
	}
	Cleanup(context, commandQueue, program, kernel);

	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	//ʹ��CPU����
	cpu_start = clock();
	for (int iX = 0; iX < ResImageW; iX++) {
		for (int iY = 0; iY < ResImageH; iY++) {
			int fHitX = iX - SampleOriginX + TranformHor;// ����λ���Խ���Ϊԭ��ĳ���λ��x
			int fHitY = iY - SampleOriginY + TranformVec; // ����λ���Խ���Ϊԭ��ĳ���λ��y
			float fHitPointNorm = sqrt((float)(fHitX*fHitX + fHitY * fHitY)); // �ཻ�����												  // ��ӳ�����ڸ�Բ����
			if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON && fHitPointNorm < SectorRadiusPixel + cuEPSILON) {
				float fSamplePointAngle = acos((float)(fHitY / fHitPointNorm - cuEPSILON)) * matlab_sign(fHitX); // Cos��Ƕ�
																											// ��ӳ���ĵ�ĽǶ�û�г�����Χ
				if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
					// CublicSample
					float PosX = (fSamplePointAngle - StartAngle) * (AveIntervalAngleReciprocal); // �ڼ�������
					float PosY = (fHitPointNorm - ProbeRadiusPixel) * (Ratio);  // �ڼ������ݵĵڼ�������
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
						// ��ת����õ������ڷ�Χ��
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
						// ��ת����õ������ڷ�Χ��
						if (PosXint1 >= 1 && PosXint1 < iLine - 2 && PosYint1 >= 1 && PosYint1 < iSmapleNum - 2) {
							//ȡ��ӳ�䵽ԭͼ�еĵ�����ܵ�16��������� 
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
						// ��ת����õ������ڷ�Χ��
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
						// ��ת����õ������ڷ�Χ��
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
						cout << "��Ч����" << endl;
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


//������
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
		printf("��ȡͼƬ��Ϣ����!");
		return 0;
	}

	
	// ��ȡͼƬ��ת����opencvͼƬ����
	int cnt = img->width * img->height;
	img->data = (float*)malloc(cnt * sizeof(float));
	fread(img->data, sizeof(float), cnt, file);


	IplImage* im = floatArrayToOpencvImage(img->data,img->width,img->height);
	IplImage* imsc = convert(img);
	Mat im_mat = cvarrToMat(im);
	Mat imsc_mat = cvarrToMat(imsc);

	cvShowImage("ԭͼƬ", im);
	cvShowImage("ת����ͼƬ", imsc);

	imwrite("C:/Users/Administrator/Desktop/res_image.jpg", imsc_mat);
	cvWaitKey();
	cvReleaseImage(&im);
	cvReleaseImage(&imsc);

	return 0;


}

// ��cl�ļ�����תΪ�ַ���
bool convertCLFile2String(const char *pFileName, std::string& Str)
{
	size_t        uiSize = 0;
	size_t        uiFileSize = 0;
	char        *pStr = NULL;
	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));
	if (fFile.is_open())
	{
		fFile.seekg(0, std::fstream::end);
		uiSize = uiFileSize = (size_t)fFile.tellg();  // ����ļ���С
		fFile.seekg(0, std::fstream::beg);
		pStr = new char[uiSize + 1];
		if (NULL == pStr)
		{
			fFile.close();
			return 0;
		}
		fFile.read(pStr, uiFileSize);                // ��ȡuiFileSize�ֽ�
		fFile.close();
		pStr[uiSize] = '\0';
		Str = pStr;
		delete[] pStr;
		return true;
	}

	cerr << "Error: Failed to open cl file\n:" << pFileName << endl;

	return 0;
}

//ѡ��OpenCLƽ̨������һ��������
cl_context CreateContext()
{
	cl_int errNum;//״̬
	cl_uint numPlatforms;//��������ƽ̨
	cl_platform_id firstPlatformId;//��һ������ƽ̨
	cl_context context = NULL;//OpenCL������
 //ѡ����õ�ƽ̨�еĵ�һ��
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);//��ȡƽ̨ID
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	//�õ�һ��ƽ̨����һ��OpenCL�����Ļ���
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};//�������������Խṹ��
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);//����������
	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "����������ʧ��" << endl;
		exit(-1);
	}

	return context;
}

//�����豸�������������
//���ݴ���������Ľ���
//ͨ��ָ�뷵���豸Id
//�����������
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;//״̬
	cl_device_id *devices;//�����豸
	cl_command_queue commandQueue = NULL;//�������
	size_t deviceBufferSize = -1;//�豸�����С

								 // ��ȡ�豸��������С
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Ϊ�豸���仺��ռ�
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);//��ȡ�豸��������Ϣ
	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "������������Ϣʧ��" << endl;
	}

	//ѡȡ�����豸�еĵ�һ���������������
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	*device = devices[0];//ͨ��ָ�봫����һ���豸��Ϣ
	delete[] devices;//�ͷ�
	return commandQueue;//�����������
}

//�����͹����������
//����OpenCL����������Ϣ���豸��Ϣ���˺����ļ�
//�Ӻ˺����ļ��м��غ˺������룬���뵽�����ĳ��������
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;//״̬
	cl_program program;//�������

					   //��ȡ�˺���Դ����
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();//��Դ�����string����תΪC�ַ������͡�
										   //�������ĺ�Դ���봴���������
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);

	//����������
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (errNum != CL_SUCCESS) {
		cout << errNum;
		cout << "����������ʧ��" << endl;
	}

	//���س������
	return program;
}

// �ͷ�OpenCL��Դ
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel)
{
	//�ͷ��������
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	//�ͷ��ں˶���
	if (kernel != 0)
		clReleaseKernel(kernel);

	//�ͷų������
	if (program != 0)
		clReleaseProgram(program);

	//�ͷ�������
	if (context != 0)
		clReleaseContext(context);
}


