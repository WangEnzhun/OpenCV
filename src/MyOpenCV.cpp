#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#pragma region  WINDOWS PARAMATER

#define WINDOW_WIDTH		500
#define WINDOW_HEIGHT		500

#define WINDOW_NAME_0		"Source Image"
#define WINDOW_NAME_1		"OpenCV environment test"
#define WINDOW_NAME_2		"Image show"
#define WINDOW_NAME_3		"Image erode"
#define WINDOW_NAME_4		"Image blur"
#define WINDOW_NAME_5		"Image Canny"
#define WINDOW_NAME_6		"Capture Video"
#define WINDOW_NAME_7		"Capture Carmera"
#define WINDOW_NAME_15		"Image Write"
#define WINDOW_NAME_16		"Image AddWeighted"
#define WINDOW_NAME_17		"Trackbar"
#define WINDOW_NAME_18		"Mouse Operation"
#define WINDOW_NAME_19		"Mat Use"
#define WINDOW_NAME_20		"Basic Graphics Rendering"
#define WINDOW_NAME_21		"Operate Pixel"
#define WINDOW_NAME_25		"Image Blending"
#define WINDOW_NAME_26		"Multi Channel Blending"
#define WINDOW_NAME_27		"Contrast And Bright"
#define WINDOW_NAME_28		"DFT"
#define WINDOW_NAME_29		"XMl/YAML Write and Read"
#define WINDOW_NAME_31		"BoxFilter"
#define WINDOW_NAME_32		"Blur"
#define WINDOW_NAME_33		"GaussianBlur"
#define WINDOW_NAME_35		"MedianBlur"
#define WINDOW_NAME_36		"BilateraFilter"
#define WINDOW_NAME_37_1	"ImageFilter Box"
#define WINDOW_NAME_37_2	"ImageFilter Blur"
#define WINDOW_NAME_37_3	"ImageFilter Gaussian"
#define WINDOW_NAME_37_4	"ImageFilter Median"
#define WINDOW_NAME_37_5	"ImageFilter Bilater"
#define WINDOW_NAME_40		"Dilate and Erode"
#define WINDOW_NAME_48_1	"Morphology Dilate"
#define WINDOW_NAME_48_2	"Morphology Erode"
#define WINDOW_NAME_48_3	"Morphology Open"
#define WINDOW_NAME_48_4	"Morphology Clese"
#define WINDOW_NAME_48_5	"Morphology Gradient"
#define WINDOW_NAME_48_6	"Morphology TopHat"
#define WINDOW_NAME_48_7	"Morphology BlackHat"
#define WINDOW_NAME_49		"FloodFill"
#define WINDOW_NAME_50		"FloodFill2"
#define WINDOW_NAME_51_1	"Resize Up"
#define WINDOW_NAME_51_2	"Resize Down"
#define WINDOW_NAME_54		"Pyr And Resize"
#define WINDOW_NAME_55		"Threshold"
#define WINDOW_NAME_56		"Canny"
#define WINDOW_NAME_57		"Sobel"
#define WINDOW_NAME_57_x	"Sobel X"
#define WINDOW_NAME_57_y	"Sobel Y"
#define WINDOW_NAME_58		"Laplacian"
#define WINDOW_NAME_59_x	"Scharr X"
#define WINDOW_NAME_59_y	"Scharr Y"
#define WINDOW_NAME_59  	"Scharr"
#define WINDOW_NAME_60_1	"Canny Edge Detection"
#define WINDOW_NAME_60_2	"Sobel Edge Detection"
#define WINDOW_NAME_61		"HoughLines"
#define	WINDOW_NAME_62		"HoughLinesP"

#pragma endregion

#pragma region Program Global Variable

Mat g_srcImage;
Mat	g_dstImage;
Mat g_dstImage1;
Mat g_dstImage2;
Mat g_dstImage3;
Mat g_dstImage4;
Mat g_dstImage5;
Mat g_dstImage6;
Mat g_dstImage7;
Mat g_tmpImage;
Mat g_grayImage;
Mat g_roiImage;
Mat g_logoImage;
Mat g_maskImage;
Mat g_gradImage_x;
Mat g_gradImage_y;

int g_nMaxTrackbarValue;
int g_nSliderTrackbarValue;	

int g_nContrastValue;
int g_nBrightValue;

int g_nBoxFilterValue = 3;
int g_nBlurValue = 3;
int g_nGaussianBlurValue = 3;
int g_nMedianValue = 3;
int g_nBilateralValue = 3;

int g_nDilateErodeTypeValue = 0;
int g_nStructElementSizeValue = 5;

int g_nElementSizeValue = 3;

int g_FillMode = 1;
int g_nLowDifference = 20;
int g_nUpDifference = 20;
int g_nConnectivity = 4;
int g_bIsColor = true;
int g_bUserMask = false;
int g_nNewMaskVal = 255;

int g_nThresholdValue = 100;
int g_nThresholdTypeValue = 0;

int g_nCannyLowThresholdValue = 1;
int g_nCannyHighThresholdValue = 3;
int g_nSobelKernelSizeValue = 1;

#pragma endregion

#pragma region Function declare

void Test();
void ShowHelpText();
void WritePixelToFile(Mat inputImage, string fileName);

void P0_LoadSourceImage();
void P1_EnvironmentTest();
void P2_ImageShow();
void P3_ImageErode();
void P4_ImageBlur();
void P5_ImageCanny();
void P6_CaptureVideo();
void P7_CaptureCarmera();
void P15_ImageWrite();
void P16_ImageAddWeight();
void P17_Trackbar();
void on_Trackbar_P17(int, void*);
void P18_MouseOperation();
void on_MouseHandle_P18(int event, int x, int y, int flags, void* param);
void P19_MatUsed();
void P20_BasicGraphicsRendering();
void DrawLine(Mat img, Point start, Point end);
void DrawRectangle(Mat img, Point p1, Point p2);
void DrawCircle(Mat img, Point center, double radius);
void DrawEllipse(Mat img, Point center, Size size, double angle);
void P21_OperatePixel();
void ColorReduce_Pointer(Mat &inputImage, Mat & outputImage, int div);
void ColorReduce_Iterator(Mat &inputImage, Mat & outputImage, int div);
void ColorReduce_DynamicAddress(Mat &inputImage, Mat & outputImage, int div);
void P25_ImageBlending();
void P26_MultiChannelBlending();
void P27_ContrastAndBright();
void on_ConstrastAndBright_P27(int, void*);
void P28_DFT();
void P29_WriteReadXMLYAML();
void P31_BoxFilter();
void P32_Blur();
void P33_GaussianBlur();
void P35_MedianBlur();
void P36_BialteraFilter();
void P37_ImageFilter();
void on_BoxFilter(int, void*);
void on_Blur(int, void*);
void on_GaussianBlur(int, void*);
void on_MedianBlur(int, void*);
void on_BilateralFilter(int, void*);
void P40_DilateAndErode();
void on_DilateAndErodeType(int, void*);
void on_ElementSize(int, void*);
void Process();
void P48_Morphology();
void on_Dilate(int, void*);
void on_Erode(int, void*);
void on_Open(int, void*);
void on_Clese(int, void *);
void on_Gradient(int, void*);
void on_TopHat(int, void*);
void on_BlackHat(int, void*);
void P49_FloodFill();
void P50_FloodFill2();
void on_MouseHandle_P50(int event,int x,int y,int,void*);
void P51_Resize();
void P54_PyrAndResize();
void P55_Threshold();
void on_Threshold(int, void*);
void P56_Canny();
void P57_Sobel(); 
void P58_Laplacian();
void P59_Scharr();
void P60_EdgeDetection();
void on_Canny(int, void*);
void on_Sobel(int, void*);
void P61_HoughLines();

#pragma endregion

int main()
{
	system("cls");
	system("color 3F");
	ShowHelpText();

	P0_LoadSourceImage();

	int key = 0;
	while (1)
	{
		key = waitKey(10);

		switch (key)
		{
		case 27:	//[ESC] for Exit
			return 0;
			break;
		case '`':	//[`] for Test
			Test();
			break;
		case '.':	//[.] for Clear Console show
			system("cls");
			ShowHelpText();
			break;
		case '0':	//[0] for Clear and Load source image
			destroyAllWindows();
			P0_LoadSourceImage();
			break;
		case '1':	//[1] for environment test
			P1_EnvironmentTest();
			break;
		case '2':	//[2] for Image Show
			P2_ImageShow();
			break;
		case '3':	//[2] for Image Erode
			P3_ImageErode();
			break;
		case '4':	//[4] for Image Blur
			P4_ImageBlur();
			break;
		case '5':	//[5] for Image Canny
			P5_ImageCanny();
			break;
		case '6':	//[6] for Capture Video
			 P6_CaptureVideo();
			 break;
		case '7':	//[7] for Capture Carmera
			P7_CaptureCarmera();
			break;
		case '8':	//[8] for Image Write
			P15_ImageWrite();
			break;
		case '9':	//[9] for Image AddWeighted
			P16_ImageAddWeight();
			break;
		case 'a':	//[a] for Trackbar
			P17_Trackbar();
			break;
		case 'b':	//[b] for Mouse Operation
			P18_MouseOperation();
			break;
		case 'c':	//[c] for Mat Used
			P19_MatUsed();
			break;
		case 'd':	//[d] for Basic Graphics Rendering
			P20_BasicGraphicsRendering();
			break;
		case 'e':	//[e] for Operate Pixel
			P21_OperatePixel();
			break;
		case 'f':	//[f] for Image Blending
			P25_ImageBlending();
			break;
		case 'g':	//[g] for Multi Channel Blending
			P26_MultiChannelBlending();
			break;
		case 'h':	//[h] for Contrast And Bright
			P27_ContrastAndBright();
			break;
		case 'i':	//[i] for DFT
			P28_DFT();
			break;
		case 'j':	//[j] for XMl/YAML Write and Read
			P29_WriteReadXMLYAML();
			break;
		case 'k':	//[k] for boxFilter
			P31_BoxFilter();
			break;
		case 'l':	//[l] for Blur
			P31_BoxFilter();
			break;
		case 'm':	//[m] for GaussianBlur
			P33_GaussianBlur();
			break;
		case 'n':	//[n] for Median Blur
			P35_MedianBlur();
			break;
		case 'o':	//[o] for Bialtera Filter
			P36_BialteraFilter();
			break;
		case 'p':	//[p] for Image Filter
			P37_ImageFilter();
			break;
		case 'q':	//[q] for Image Dilate and Erode
			P40_DilateAndErode();
			break;
		case 'r':	//[r] for Morphology
			P48_Morphology();
			break;
		case 's':	//[s] for FloodFill
			P49_FloodFill();
			break;
		case 't':	//[t] for FloodFill2
			P50_FloodFill2();
			break;
		case 'u':	//[u] for Resize
			P51_Resize();
			break;
		case 'v':	//[v] for Pyr and Resize
			P54_PyrAndResize();
			break;
		case 'w':	//[w] for Threshold
			P55_Threshold();
			break;
		case 'x':	//[x] for Canny
			P56_Canny();
			break;
		case 'y':	//[y] for Sobel
			P57_Sobel();
			break;
		case 'z':	//[z] for Laplacian
			P58_Laplacian();
			break;
		case 'A':	//[A] for Scharr
			P59_Scharr();
			break;
		case 'B':	//[B] for Edge Detection
			P60_EdgeDetection();
			break;
		case 'C':	//[C] for HoughLines
			P61_HoughLines();
			break;
		case 'D':	//[D] for 

			break;
		case 'E':	//[E] for 

			break;
		case 'F':	//[F] for 

			break;
		case 'G':	//[G] for 

			break;
		case 'H':	//[H] for 

			break;
		case 'I':	//[I] for 

			break;
		case 'J':	//[J] for 

			break;
		case 'K':	//[K] for 

			break;
		case 'L':	//[L] for 

			break;
		case 'M':	//[M] for 

			break;
		case 'N':	//[N] for 

			break;
		case 'O':	//[O] for 

			break;
		case 'P':	//[P] for 

			break;
		case 'Q':	//[Q] for 

			break;
		case 'R':	//[R] for 

			break;
		case 'S':	//[S] for 

			break;
		case 'T':	//[T] for 

			break;
		case 'U':	//[U] for 

			break;
		case 'V':	//[V] for 

			break;
		case 'W':	//[W] for 

			break;
		case 'X':	//[X] for 

			break;
		case 'Y':	//[Y] for 

			break;
		case 'Z':	//[Z] for 

			break;

		default:
			break;
		}
	}
	return 0;
}

#pragma region Function Definition

void WritePixelToFile(Mat inputImage, string fileName)
{
	int nRows = inputImage.rows;
	int nCols = inputImage.cols;
	int nChannels = inputImage.channels();
	int nType = inputImage.type();

	fstream file(fileName, ios::out);
	file << "Rows=" << nRows << "," << "Cols=" << nCols << "," << "Channels=" << nChannels << endl;
	for (int row = 0; row < nRows; row++)
	{
		for (int col = 0; col < nCols; col++)
		{
			if (nChannels == 1)
			{
				file << (int)inputImage.at<uchar>(row, col) << ",";
			}
			else if (nChannels == 2)
			{
				Vec2i YN_P = inputImage.at<Vec2b>(row, col);
				file << YN_P.val[0] << "," << YN_P.val[1] << ",";
			}
			else if (nChannels == 3)
			{
				Vec3i RGB_P = inputImage.at<Vec3b>(row, col);
				file << RGB_P.val[0] << "," << RGB_P.val[1] << "," << RGB_P.val[2] << ",";
			}
			else if (nChannels == 4)
			{
				Vec4i RGBA_P = inputImage.at<Vec4b>(row, col);
				file << RGBA_P.val[0] << "," << RGBA_P.val[1] << "," << RGBA_P.val[2] << "," << RGBA_P.val[3] << ",";
			}
		}
		file << endl;
	}
	file.close();
	cout << "Rows=" << nRows << "\t" << "Cols=" << nCols << "\t" << "Channels=" << nChannels << endl;
	cout << fileName <<" Write Finished" << endl;
}
void ShowHelpText()
{
	printf("\n\n\t\t\t  The OpenCV Version:" CV_VERSION);
	printf("\n\n  ----------------------------------------------------------------------------\n");
	printf("\n\tWelcome to OpenCV Learning Program\n\n");
	printf("\n\n\tKeyBoard Operation: \n\n"
		"\t\tKeyBoard[Esc]- Program    for Exit\n"
		"\t\tKeyBoard[`]  - Program    for Test\n"
		"\t\tKeyBoard[.]  - Program    for Clear Console show\n"
		"\t\tKeyBoard[0]  - Program 0  for Load Soutce Image\n"
		"\t\tKeyBoard[1]  - Program 1  for Environment test\n"
		"\t\tKeyBoard[2]  - Program 2  for Image Show\n"
		"\t\tKeyBoard[3]  - Program 3  for Image Erode\n"
		"\t\tKeyBoard[4]  - Program 4  for Image Blur\n"
		"\t\tKeyBoard[5]  - Program 5  for Image Cannny\n"
		"\t\tKeyBoard[6]  - Program 6  for Capture Video-must press[Enter] for break\n"
		"\t\tKeyBoard[7]  - Program 7  for Capture Carmera-must press[Enter] for break\n"
		"\t\tKeyBoard[8]  - Program 15 for Image Write\n"
		"\t\tKeyBoard[9]  - Program 16 for Image AddWeighted\n"
		"\t\tKeyBoard[a]  - Program 17 for Trackbar\n"
		"\t\tKeyBoard[b]  - Program 18 for Mouse Operation\n"
		"\t\tKeyBoard[c]  - Program 19 for Mat Used\n"
		"\t\tKeyBoard[d]  - Program 20 for Basic Graphics Rendering\n"
		"\t\tKeyBoard[e]  - Program 21 for Operate Pixel\n"
		"\t\tKeyBoard[f]  - Program 25 for Image Blending\n"
		"\t\tKeyBoard[g]  - Program 26 for Multi Channel Blending\n"
		"\t\tKeyBoard[h]  - Program 27 for Contrast And Bright\n"
		"\t\tKeyBoard[i]  - Program 28 for DFT\n"
		"\t\tKeyBoard[j]  - Program 29 for XMl/YAML Write and Read\n"
		"\t\tKeyBoard[k]  - Program 31 for Box Filter\n"
		"\t\tKeyBoard[l]  - Program 32 for Blur\n"
		"\t\tKeyBoard[m]  - Program 33 for Gaussina Blur\n"
		"\t\tKeyBoard[n]  - Program 35 for Median Blur\n"
		"\t\tKeyBoard[o]  - Program 36 for Bialtera Filter\n"
		"\t\tKeyBoard[p]  - Program 37 for Image Filter\n"
		"\t\tKeyBoard[q]  - Program 40 for Image Dilate and Erode\n"
		"\t\tKeyBoard[r]  - Program 48 for Morphology\n"
		"\t\tKeyBoard[s]  - Program 49 for Flood Fill\n"
		"\t\tKeyBoard[t]  - Program 50 for Flood Fill 2\n"
		"\t\tKeyBoard[u]  - Program 51 for Resize\n"
		"\t\tKeyBoard[v]  - Program 54 for Pyr and Resize\n"
		"\t\tKeyBoard[w]  - Program 55 for Threshold\n"
		"\t\tKeyBoard[x]  - Program 56 for Canny\n"
		"\t\tKeyBoard[y]  - Program 57 for Sobel\n"
		"\t\tKeyBoard[z]  - Program 58 for Laplacian\n"
		"\t\tKeyBoard[A]  - Program 59 for Scharr\n"
		"\t\tKeyBoard[B]  - Program 60 for Edge Detection\n"
		"\t\tKeyBoard[C]  - Program 61 for HoughLines\n"
		//"\t\tKeyBoard[D]  - Program XX for \n"
		//"\t\tKeyBoard[E]  - Program XX for \n"
		//"\t\tKeyBoard[F]  - Program XX for \n"
		//"\t\tKeyBoard[G]  - Program XX for \n"
		//"\t\tKeyBoard[H]  - Program XX for \n"
		//"\t\tKeyBoard[I]  - Program XX for \n"
		//"\t\tKeyBoard[J]  - Program XX for \n"
		//"\t\tKeyBoard[K]  - Program XX for \n"
		//"\t\tKeyBoard[L]  - Program XX for \n"
		//"\t\tKeyBoard[M]  - Program XX for \n"
		//"\t\tKeyBoard[N]  - Program XX for \n"
		//"\t\tKeyBoard[O]  - Program XX for \n"
		//"\t\tKeyBoard[P]  - Program XX for \n"
		//"\t\tKeyBoard[Q]  - Program XX for \n"
		//"\t\tKeyBoard[R]  - Program XX for \n"
		//"\t\tKeyBoard[S]  - Program XX for \n"
		//"\t\tKeyBoard[T]  - Program XX for \n"
		//"\t\tKeyBoard[U]  - Program XX for \n"
		//"\t\tKeyBoard[V]  - Program XX for \n"
		//"\t\tKeyBoard[W]  - Program XX for \n"
		//"\t\tKeyBoard[X]  - Program XX for \n"
		//"\t\tKeyBoard[Y]  - Program XX for \n"
		//"\t\tKeyBoard[Z]  - Program XX for \n"
		"\n\n\n"
		
	);
}
void P0_LoadSourceImage()
{
	g_srcImage = imread("lena.jpg");
	namedWindow(WINDOW_NAME_0, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME_0, g_srcImage);
}
void P1_EnvironmentTest()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage = g_srcImage;
	namedWindow(WINDOW_NAME_1);
	imshow(WINDOW_NAME_1, g_dstImage);
}
void P2_ImageShow()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage = g_srcImage;
	namedWindow(WINDOW_NAME_2);
	imshow(WINDOW_NAME_2, g_dstImage);
}
void P3_ImageErode()
{
	g_srcImage = imread("lena.jpg");
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(g_srcImage, g_dstImage, element);
	namedWindow(WINDOW_NAME_3);
	imshow(WINDOW_NAME_3, g_dstImage);
}
void P4_ImageBlur()
{
	g_srcImage = imread("lena.jpg");
	blur(g_srcImage, g_dstImage, Size(5, 5));
	namedWindow(WINDOW_NAME_4);
	imshow(WINDOW_NAME_4, g_dstImage);
}
void P5_ImageCanny()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage.create(g_srcImage.size(), g_dstImage.type());
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	blur(g_grayImage, g_dstImage, Size(3, 3));
	Canny(g_dstImage, g_dstImage, 10, 30);
	namedWindow(WINDOW_NAME_5);
	imshow(WINDOW_NAME_5, g_dstImage);
}
void P6_CaptureVideo()
{
	VideoCapture capture("dota.avi");
	while (1)
	{
		capture >> g_dstImage;
		imshow(WINDOW_NAME_6, g_dstImage);
		int exitKey=waitKey(30);
		if (exitKey == 13) break;
	}
}
void P7_CaptureCarmera()
{
	VideoCapture capture(0);
	while (1)
	{
		Mat frame;
		capture >> frame;

		Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
		erode(frame, g_dstImage, element);

		namedWindow(WINDOW_NAME_7);
		imshow(WINDOW_NAME_7, g_dstImage);
		int exitKey = waitKey(30);
		if (exitKey == 13) break;
	}
}
void P15_ImageWrite()
{
	g_srcImage = imread("lena.jpg");
	cvtColor(g_srcImage, g_dstImage, COLOR_BGR2GRAY);
	namedWindow(WINDOW_NAME_15);
	imshow(WINDOW_NAME_15, g_dstImage);
	imwrite("lena_copy.jpg", g_dstImage);
}
void P15_ImageWrite_Alpha()
{
	//Mat mat(480, 640, CV_8UC4);
	//for (int i = 0; i < mat.rows; ++i)
	//{
	//	for (int j = 0; j < mat.cols; ++j)
	//	{
	//		mat.at<Vec4b>(i, j)[0] = rand() % 255;
	//		mat.at<Vec4b>(i, j)[1] = rand() % 255;
	//		mat.at<Vec4b>(i, j)[2] = rand() % 255;
	//		mat.at<Vec4b>(i, j)[3] = rand() % 255;
	//	}
	//}
	
	//Mat mat(480, 640, CV_8UC1);
	//for (int i = 0; i < mat.rows; ++i)
	//{
	//	for (int j = 0; j < mat.cols; ++j)
	//	{
	//		mat.at<uchar>(i, j) = rand() % 255;
	//	}
	//}

	Mat mat(480, 640, CV_8UC3);
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			mat.at<Vec3b>(i, j)[0] = saturate_cast<uchar>((float(mat.cols-j))/((float)mat.cols)*UCHAR_MAX);
			mat.at<Vec3b>(i, j)[1] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows)*UCHAR_MAX);
			mat.at<Vec3b>(i, j)[2] = UCHAR_MAX;//saturate_cast<uchar>(0.5*(mat.at<Vec3b>(i, j)[0] + mat.at<Vec3b>(i, j)[1]));
		}
	}
	namedWindow(WINDOW_NAME_15);
	imshow(WINDOW_NAME_15, mat);
	imwrite("dota_copy.jpg", mat);
}
void P16_ImageAddWeight()
{
	g_srcImage = imread("dota.jpg");
	g_logoImage = imread("dota_logo.jpg");

	g_roiImage = g_srcImage(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));
	addWeighted(g_roiImage, 0.2, g_logoImage, 0.8, 0, g_roiImage);

	namedWindow(WINDOW_NAME_16);
	imshow(WINDOW_NAME_16, g_srcImage);

}
void P17_Trackbar()
{
	
	g_nSliderTrackbarValue = 80;
	g_nMaxTrackbarValue = 100;

	namedWindow(WINDOW_NAME_17);
	createTrackbar(WINDOW_NAME_17, WINDOW_NAME_17,&g_nSliderTrackbarValue,g_nMaxTrackbarValue,on_Trackbar_P17);
	on_Trackbar_P17(g_nSliderTrackbarValue, 0);

}
void on_Trackbar_P17(int, void*)
{
	double dAlphaValue = (double)g_nSliderTrackbarValue / g_nMaxTrackbarValue;
	double dBetaValue = 1.0 - dAlphaValue;

	g_srcImage = imread("dota.jpg");
	g_logoImage = imread("dota_logo.jpg");
	g_roiImage = g_srcImage(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));

	addWeighted(g_roiImage, dAlphaValue, g_logoImage, dBetaValue, 0, g_roiImage);
	imshow(WINDOW_NAME_17, g_srcImage);
	waitKey(30);
}
void P18_MouseOperation()
{
	g_srcImage = imread("lena.jpg");
	g_srcImage.copyTo(g_tmpImage);
	g_srcImage.copyTo(g_dstImage);
	namedWindow(WINDOW_NAME_18);
	setMouseCallback(WINDOW_NAME_18, on_MouseHandle_P18, 0);
	imshow(WINDOW_NAME_18, g_tmpImage);

}
void on_MouseHandle_P18(int event, int x, int y, int flags, void* param)
{
	static Point pre_pt = (-1, -1);
	static Point cur_pt = (-1, -1);
	char temp[16];
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		g_srcImage.copyTo(g_dstImage);
		sprintf_s(temp, "(%d,%d)", x, y);
		pre_pt = Point(x, y);
		putText(g_dstImage, temp, pre_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255), 1, 8);
		circle(g_dstImage, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		imshow(WINDOW_NAME_18, g_dstImage);
	}
	else if (event == CV_EVENT_MOUSEMOVE && !(flags&CV_EVENT_FLAG_LBUTTON))
	{
		g_dstImage.copyTo(g_tmpImage);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(g_tmpImage, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
		imshow(WINDOW_NAME_18, g_tmpImage);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
	{
		g_dstImage.copyTo(g_tmpImage);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(g_tmpImage, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
		rectangle(g_tmpImage, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);
		imshow(WINDOW_NAME_18, g_tmpImage);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		g_srcImage.copyTo(g_dstImage);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(g_dstImage, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
		circle(g_dstImage, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		rectangle(g_tmpImage, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);
		imshow(WINDOW_NAME_18, g_dstImage);
		g_dstImage.copyTo(g_tmpImage);

		int width = abs(pre_pt.x - cur_pt.x);
		int height = abs(pre_pt.y - cur_pt.y);
		if (width == 0 || height == 0)
		{
			//printf("width==0||height==0");
			return ;
		}
		Mat cutImage = g_srcImage(Rect(min(cur_pt.x, pre_pt.x), min(cur_pt.y, pre_pt.y), width, height));
		namedWindow("cutImage");
		imshow("cutImage", cutImage);
	}
}
void P19_MatUsed()
{
	cout << "------------------P19 Mat Used-----------------------" << endl;
	//Init Mat 1
	Mat M1(2, 2, CV_8UC3, Scalar(0,0,255));
	cout << "M1= " << endl << M1 << endl;
	//Init Mat 2
	Mat M2;
	M2.create(4, 4, CV_8UC3);
	cout << "M2= " << endl << M2 << endl;
	// Init Mat T
	Mat E = Mat::eye(4, 4, CV_64F);
	cout << "E= " << endl << E << endl;
	Mat O = Mat::ones(3, 3, CV_32F);
	cout << "O= " << endl << O << endl;
	Mat Z = Mat::zeros(2, 2, CV_8UC3);
	cout << "Z= " << endl << Z << endl;
	//Init Mat 3
	Mat M3 = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	cout << "M3= " << endl << M3 << endl;
	//Init Mat 4
	Mat M4 = M1.row(1).clone();
	cout << "M4= " << endl << M4 << endl<<endl;

	//Point 2
	Point2f p2f(6, 2);
	cout << "[2f]p= " << endl << p2f  << endl;
	//Point 3
	Point3f p3f(8, 2, 0);
	cout << "[3f]p= " << endl << p3f << endl;
	//Mat std::vector
	vector<float> v;
	v.push_back(3);
	v.push_back(5);
	v.push_back(7);
	cout << "[std::vector]v = " << endl << Mat(v) << endl;
	//Mat std::vector
	vector<Point2f> points(20);
	for (size_t i = 0; i < points.size(); ++i)
		points[i] = Point2f((float)(i * 5), (float)(i % 7));
	cout << "[2f]points = " << endl << points << ";" << endl;
	cout << "-----------------------------------------------------" << endl<<endl;
}
void P20_BasicGraphicsRendering()
{
	Mat atomImage = Mat::zeros(WINDOW_WIDTH, WINDOW_HEIGHT, CV_8UC3);

	Point2f ellipseCenter(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
	Size ellipseSize(WINDOW_WIDTH / 4, WINDOW_HEIGHT / 16);
	DrawEllipse(atomImage, ellipseCenter, ellipseSize, 0);
	DrawEllipse(atomImage, ellipseCenter, ellipseSize, 45);
	DrawEllipse(atomImage, ellipseCenter, ellipseSize, 90);
	DrawEllipse(atomImage, ellipseCenter, ellipseSize, 135);

	Point2f circleCenter(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
	double circleRadius = WINDOW_WIDTH / 32;
	DrawCircle(atomImage, circleCenter, circleRadius);

	int len = 14;
	int num = 10;
	for (int i = 0; i < num; i++)
	{
		Point2f tl_start(i*len, i*len), tl_end((i + 1)*len, (i + 1)*len);
		DrawRectangle(atomImage, tl_start, tl_end);
		Point2f tr_start(i*len, WINDOW_WIDTH-i*len), tr_end((i + 1)*len, WINDOW_WIDTH-(i + 1)*len);
		DrawRectangle(atomImage, tr_start, tr_end);
		Point2f bl_start(WINDOW_HEIGHT-i*len, i*len), bl_end(WINDOW_HEIGHT-(i + 1)*len, (i + 1)*len);
		DrawRectangle(atomImage, bl_start, bl_end);
		Point2f br_start(WINDOW_HEIGHT-i*len, WINDOW_WIDTH-i*len), br_end(WINDOW_HEIGHT-(i + 1)*len, WINDOW_WIDTH-(i + 1)*len);
		DrawRectangle(atomImage, br_start, br_end);
	}

	//Point2f tl(0, 0), tr(0, WINDOW_WIDTH), bl(WINDOW_HEIGHT, 0), br(WINDOW_HEIGHT, WINDOW_WIDTH);
	Point2f tl(WINDOW_HEIGHT/4-5, WINDOW_WIDTH/4-5), tr(WINDOW_HEIGHT/4-5, WINDOW_WIDTH/4*3+5), bl(WINDOW_HEIGHT/4*3+5, WINDOW_WIDTH/4-5), br(WINDOW_HEIGHT/4*3+5, WINDOW_WIDTH/4*3+5);
	DrawLine(atomImage, tl, tr);
	DrawLine(atomImage, tr, br);
	DrawLine(atomImage, br, bl);
	DrawLine(atomImage, bl, tl);

	namedWindow(WINDOW_NAME_20);
	imshow(WINDOW_NAME_20, atomImage);
}
void DrawLine(Mat img, Point start, Point end)
{
	int thickness = 2;
	int lineType = 8;
	line(img,
		start,
		end,
		Scalar(0, 0, 255),
		thickness,
		lineType);
}
void DrawRectangle(Mat img, Point p1, Point p2)
{
	int thickness = -1;
	int lineType = 8;
	rectangle(img, p1, p2, Scalar(0, 255, 0), thickness, lineType, 0);
}
void DrawCircle(Mat img, Point center, double radius)
{
	int thickness = -1;
	int lineType = 8;
	circle(img,
		center,
		radius,
		Scalar(255, 0, 0),
		thickness,
		lineType);
}
void DrawEllipse(Mat img,Point center,Size size, double angle)
{
	int thickness = 1;
	int lineType = 8;
	ellipse(img,
		center,
		size,
		angle,
		0,
		360,
		Scalar(255,255,255),
		thickness,
		lineType);
}
void P21_OperatePixel()
{
	cout << "------------------P19 Mat Used-----------------------" << endl;
	g_srcImage = imread("lena.jpg");
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	double time1 = static_cast<double>(getTickCount());
	ColorReduce_Pointer(g_srcImage, g_dstImage, 32);
	time1 = ((double)getTickCount() - time1) / getTickFrequency();

	double time2 = static_cast<double>(getTickCount());
	ColorReduce_Iterator(g_srcImage, g_dstImage, 32);
	time2 = ((double)getTickCount() - time2) / getTickFrequency();

	double time3 = static_cast<double>(getTickCount());
	ColorReduce_DynamicAddress(g_srcImage, g_dstImage, 32);
	time3 = ((double)getTickCount() - time3) / getTickFrequency();

	namedWindow(WINDOW_NAME_21);
	imshow(WINDOW_NAME_21, g_dstImage);

	cout << "Time1_Pointer= " << time1 << endl;
	cout << "Time2_Iterator= " << time2 << endl;
	cout << "Time3_DynamicAdress= " << time3 << endl;

	cout << "-----------------------------------------------------" << endl<<endl;
}
void ColorReduce_Pointer(Mat &inputImage, Mat & outputImage, int div)
{
	outputImage = inputImage.clone();
	int rowNumber = outputImage.rows;
	int colNumber = outputImage.cols*outputImage.channels();
	for (int i = 0; i < rowNumber; i++)
	{
		uchar*data = outputImage.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
		{
			data[j] = data[j] / div*div + div / 2;
		}
	}
}
void ColorReduce_Iterator(Mat &inputImage, Mat & outputImage, int div)
{
	outputImage = inputImage.clone();
	Mat_<Vec3b>::iterator it = outputImage.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = outputImage.end<Vec3b>();
	for (; it != itend; ++it)
	{
		(*it)[0] = (*it)[0] / div*div + div / 2;
		(*it)[1] = (*it)[1] / div*div + div / 2;
		(*it)[2] = (*it)[2] / div*div + div / 2;
	}
}
void ColorReduce_DynamicAddress(Mat &inputImage, Mat & outputImage, int div)
{
	outputImage = inputImage.clone();
	int rowNumber = outputImage.rows;
	int colNumber = outputImage.cols;
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			outputImage.at<Vec3b>(i, j)[0] = outputImage.at<Vec3b>(i, j)[0] / div*div + div / 2;
			outputImage.at<Vec3b>(i, j)[1] = outputImage.at<Vec3b>(i, j)[1] / div*div + div / 2;
			outputImage.at<Vec3b>(i, j)[2] = outputImage.at<Vec3b>(i, j)[2] / div*div + div / 2;
		}
	}
}
void P25_ImageBlending()
{
	g_srcImage = imread("dota.jpg");
	g_logoImage = imread("dota_logo.jpg");
	g_roiImage = g_srcImage(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));
	
	Mat mask = imread("dota_logo.jpg", 0);
	g_logoImage.copyTo(g_roiImage, mask);

	namedWindow(WINDOW_NAME_25);
	imshow(WINDOW_NAME_25,g_srcImage);
}
void P26_MultiChannelBlending()
{
	vector<Mat> channels;

	//--------------Blue-------------------//
	Mat  imageBlueChannel, imageBlueChannelROI;
	g_logoImage = imread("dota_logo.jpg", 0);
	g_srcImage = imread("dota.jpg");

	split(g_srcImage, channels);
	imageBlueChannel = channels.at(0);

	imageBlueChannelROI = imageBlueChannel(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));
	addWeighted(imageBlueChannelROI, 1.0,g_logoImage, 0.5, 0, imageBlueChannelROI);

	merge(channels, g_srcImage);

	imshow("ROI_Blue", g_srcImage);


	//--------------Green-------------------//
	Mat  imageGreenChannel,imageGreenChannelROI;
	g_logoImage = imread("dota_logo.jpg", 0);
	g_srcImage = imread("dota.jpg");
	
	split(g_srcImage, channels);
	imageGreenChannel = channels.at(1);

	imageGreenChannelROI = imageGreenChannel(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));
	addWeighted(imageGreenChannelROI, 1.0,g_logoImage, 0.5, 0., imageGreenChannelROI);

	merge(channels, g_srcImage);

	imshow("ROI_Green", g_srcImage);


	//--------------Red-------------------//
	Mat  imageRedChannel, imageRedChannelROI;
	g_logoImage = imread("dota_logo.jpg", 0);
	g_srcImage = imread("dota.jpg");

	split(g_srcImage, channels);
	imageRedChannel = channels.at(2);
	
	imageRedChannelROI = imageRedChannel(Rect(800, 350, g_logoImage.cols, g_logoImage.rows));
	addWeighted(imageRedChannelROI, 1.0,g_logoImage, 0.5, 0., imageRedChannelROI);

	merge(channels, g_srcImage);

	imshow("ROI_Red ", g_srcImage);
}
void P27_ContrastAndBright()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	g_nContrastValue = 80;
	g_nBrightValue = 80;
	namedWindow(WINDOW_NAME_27);
	createTrackbar("Constrast:", WINDOW_NAME_27, &g_nContrastValue, 300, on_ConstrastAndBright_P27);
	createTrackbar("Bright:", WINDOW_NAME_27, &g_nBrightValue, 200, on_ConstrastAndBright_P27);
	on_ConstrastAndBright_P27(g_nContrastValue, 0);
	on_ConstrastAndBright_P27(g_nBrightValue, 0);

}
void on_ConstrastAndBright_P27(int, void*)
{
	
	for (int i = 0; i < g_srcImage.rows; i++)
	{
		for (int j = 0; j < g_srcImage.cols; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				g_dstImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(i, j)[c]) + g_nBrightValue);
			}
		}
	}
	imshow(WINDOW_NAME_27, g_dstImage);
}
void P28_DFT()
{
	g_srcImage = imread("lena.jpg",0);
	int m = getOptimalDFTSize(g_srcImage.rows);
	int n = getOptimalDFTSize(g_srcImage.cols);
	Mat padded;
	copyMakeBorder(g_srcImage, padded, 0, m - g_srcImage.rows,0, n - g_srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);

	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);

	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols &-2, magnitudeImage.rows &-2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;

	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	namedWindow(WINDOW_NAME_28);
	imshow(WINDOW_NAME_28, magnitudeImage);

}
void P29_WriteReadXMLYAML()
{
	cout << "------------------P29 WriteReadXMLYAML-----------------------" << endl;
	//FileStorage fs_w("test.xml", FileStorage::WRITE);
	FileStorage fs_w("test.yaml", FileStorage::WRITE);

	fs_w << "Name" << "WangEnzhun";

	Mat E = Mat_<uchar>::eye(3, 3);
	fs_w << "E" << E;

	fs_w << "strings" << "[";
	fs_w << "image1.jpg" << "Awesomeones" << "baboon.jpg";
	fs_w << "]";

	fs_w << "Mapping" << "{";
	fs_w << "one" << 1;
	fs_w << "two" << 2;
	fs_w << "}";
	
	fs_w << "RandNumber" << rand() % 255;
	fs_w.release();
	cout << "Write XML/YAML file finished!"<<endl << endl;

	//FileStorage fs_r("test.xml", FileStorage::READ);
	FileStorage fs_r("test.yaml", FileStorage::READ);

	string  Name;
	fs_r["Name"] >> Name;
	Name = (string) fs_r["Name"];
	cout << "Name=" << Name << endl;

	Mat M;
	fs_r["E"] >> M;
	cout << "E=" << endl << M << endl;

	FileNode n = fs_r["strings"];
	if (n.type() != FileNode::SEQ)
	{
		return;
	}
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		cout << (string)*it << endl;
	cout<<endl << "Read XML/YAML file finished!" << endl;
	cout << "-----------------------------------------------------" << endl << endl;
}
void P31_BoxFilter()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	boxFilter(g_srcImage, g_dstImage, -1, Size(5, 5));

	namedWindow(WINDOW_NAME_31);
	imshow(WINDOW_NAME_31, g_dstImage);
}
void P32_Blur()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	blur(g_srcImage, g_dstImage, Size(9, 9));

	namedWindow(WINDOW_NAME_32);
	imshow(WINDOW_NAME_32, g_dstImage);
}
void P33_GaussianBlur()
{
	g_srcImage = imread("lena.jpg");
	
	GaussianBlur(g_srcImage, g_dstImage, Size(5, 5), 0, 0);

	namedWindow(WINDOW_NAME_33);
	imshow(WINDOW_NAME_33, g_dstImage);
}
void P35_MedianBlur()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage = g_srcImage.clone();
	medianBlur(g_srcImage, g_dstImage,25);
	namedWindow(WINDOW_NAME_35);
	imshow(WINDOW_NAME_35, g_dstImage);

}
void P36_BialteraFilter()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage = g_srcImage.clone();
	bilateralFilter(g_srcImage, g_dstImage, 25, 25*2, 25/2);
	namedWindow(WINDOW_NAME_36);
	imshow(WINDOW_NAME_36, g_dstImage);

}
void P37_ImageFilter()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage1 = g_srcImage.clone();
	g_dstImage2 = g_srcImage.clone();
	g_dstImage3 = g_srcImage.clone();
	g_dstImage4 = g_srcImage.clone();
	g_dstImage5 = g_srcImage.clone();

	namedWindow(WINDOW_NAME_37_1);
	createTrackbar("LineFilter_Box:", WINDOW_NAME_37_1, &g_nBoxFilterValue, 50, on_BoxFilter);
	on_BoxFilter(g_nBoxFilterValue, 0);

	namedWindow(WINDOW_NAME_37_2);
	createTrackbar("LineFilter_Blur:", WINDOW_NAME_37_2, &g_nBlurValue, 50, on_Blur);
	on_BoxFilter(g_nBlurValue, 0);

	namedWindow(WINDOW_NAME_37_3);
	createTrackbar("LineFilter_Guassian:", WINDOW_NAME_37_3, &g_nGaussianBlurValue, 50, on_GaussianBlur);
	on_BoxFilter(g_nGaussianBlurValue, 0);

	namedWindow(WINDOW_NAME_37_4);
	createTrackbar("UlineFilter_Median:", WINDOW_NAME_37_4, &g_nMedianValue, 50, on_MedianBlur);
	on_MedianBlur(g_nMedianValue, 0);

	namedWindow(WINDOW_NAME_37_5);
	createTrackbar("UlineFilter_Bialter:", WINDOW_NAME_37_5, &g_nBilateralValue, 50, on_BilateralFilter);
	on_BilateralFilter(g_nBilateralValue, 0);
}
void on_BoxFilter(int, void*)
{
	boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	imshow(WINDOW_NAME_37_1, g_dstImage1);
}
void on_Blur(int, void*)
{
	blur(g_srcImage, g_dstImage2, Size(g_nBlurValue + 1, g_nBlurValue + 1));
	imshow(WINDOW_NAME_37_2, g_dstImage2);
}
void on_GaussianBlur(int, void*)
{
	GaussianBlur(g_srcImage, g_dstImage3, Size(2 * g_nGaussianBlurValue + 1, 2 * g_nGaussianBlurValue + 1), 0, 0);
	imshow(WINDOW_NAME_37_3, g_dstImage3);
}
void on_MedianBlur(int, void*)
{
	medianBlur(g_srcImage, g_dstImage4, g_nMedianValue * 2 + 1);
	imshow(WINDOW_NAME_37_4, g_dstImage4);
}
void on_BilateralFilter(int, void*)
{	
	bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralValue, g_nBilateralValue * 2, g_nBilateralValue / 2);
	imshow(WINDOW_NAME_37_5,g_dstImage5);
}
void P40_DilateAndErode()
{
	g_srcImage = imread("lena.jpg");
	namedWindow(WINDOW_NAME_40);

	createTrackbar("dilate/erode:", WINDOW_NAME_40, &g_nDilateErodeTypeValue, 1, on_DilateAndErodeType);
	createTrackbar("element size:", WINDOW_NAME_40, &g_nStructElementSizeValue, 25, on_ElementSize);
	on_DilateAndErodeType(g_nDilateErodeTypeValue, 0);
	on_ElementSize(g_nStructElementSizeValue, 0);
}
void on_DilateAndErodeType(int, void*)
{
	Process();
}
void on_ElementSize(int, void*)
{
	Process();
}
void Process() 
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSizeValue + 1, 2 * g_nStructElementSizeValue + 1), Point(g_nStructElementSizeValue, g_nStructElementSizeValue));
	if (g_nDilateErodeTypeValue == 0)
	{
		dilate(g_srcImage, g_dstImage, element);
	}
	else
	{
		erode(g_srcImage, g_dstImage, element);
	}
	imshow(WINDOW_NAME_40, g_dstImage);
}
void P48_Morphology()
{
	//dst = dilate(src, element)
	//dst = erode(src, element)
	//dst = open(src, element) = dilate(erode(src, element))
	//dst = clese(src, element) = erode(dilate(src, element))
	//dst = morph - grad(src, element) = dilate(src, element) - erode(src, element)
	//dst = tophat(src, element) = src - open(src, element)
	//dst = blackhat(src, element) = clese(src, element) - src
	g_srcImage = imread("lena.jpg");
	g_dstImage1 = g_srcImage.clone();
	g_dstImage2 = g_srcImage.clone();
	g_dstImage3 = g_srcImage.clone();
	g_dstImage4 = g_srcImage.clone();
	g_dstImage5 = g_srcImage.clone();
	g_dstImage6 = g_srcImage.clone();
	g_dstImage7 = g_srcImage.clone();

	namedWindow(WINDOW_NAME_48_1);
	createTrackbar("dilate:", WINDOW_NAME_48_1, &g_nElementSizeValue, 25,on_Dilate);
	on_Dilate(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_2);
	createTrackbar("erode:", WINDOW_NAME_48_2, &g_nElementSizeValue, 25, on_Erode);
	on_Erode(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_3);
	createTrackbar("open:", WINDOW_NAME_48_3, &g_nElementSizeValue, 25, on_Open);
	on_Open(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_4);
	createTrackbar("clese:", WINDOW_NAME_48_4, &g_nElementSizeValue, 25, on_Clese);
	on_Clese(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_5);
	createTrackbar("morph:", WINDOW_NAME_48_5, &g_nElementSizeValue, 25, on_Gradient);
	on_Gradient(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_6);
	createTrackbar("tophat:", WINDOW_NAME_48_6, &g_nElementSizeValue, 25, on_TopHat);
	on_TopHat(g_nElementSizeValue, 0);
	namedWindow(WINDOW_NAME_48_7);
	createTrackbar("blackhat:", WINDOW_NAME_48_7, &g_nElementSizeValue, 25, on_BlackHat);
	on_BlackHat(g_nElementSizeValue, 0);

}
void on_Dilate(int, void*) 
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage1, MORPH_DILATE, element);
	imshow(WINDOW_NAME_48_1, g_dstImage1);

}
void on_Erode(int, void*) 
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage2, MORPH_ERODE, element);
	imshow(WINDOW_NAME_48_2, g_dstImage2);
}
void on_Open(int, void*)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage3, MORPH_OPEN, element);
	imshow(WINDOW_NAME_48_3, g_dstImage3);
}
void on_Clese(int, void *)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage4, MORPH_CLOSE, element);
	imshow(WINDOW_NAME_48_4, g_dstImage4);
}
void on_Gradient(int, void*) 
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage5, MORPH_GRADIENT, element);
	imshow(WINDOW_NAME_48_5, g_dstImage5);
}
void on_TopHat(int, void*)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage6, MORPH_TOPHAT, element);
	imshow(WINDOW_NAME_48_6, g_dstImage6);
}
void on_BlackHat(int, void*)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nElementSizeValue + 1, 2 * g_nElementSizeValue + 1), Point(g_nElementSizeValue, g_nElementSizeValue));
	morphologyEx(g_srcImage, g_dstImage7, MORPH_BLACKHAT, element);
	imshow(WINDOW_NAME_48_7, g_dstImage7);
}
void P49_FloodFill()
{
	g_srcImage = imread("lena.jpg");
	Rect ccomp;
	floodFill(g_srcImage, Point(30, 200), Scalar(255, 255, 255), &ccomp, Scalar(10, 10, 10), Scalar(10, 10, 10));
	namedWindow(WINDOW_NAME_49);
	imshow(WINDOW_NAME_49, g_srcImage);
}
void P50_FloodFill2()
{
	destroyAllWindows();
	system("cls");
	system("color 2F");
	printf("\n\n\n\tWelcome to FloodFill\n\n");
	printf("\n\n\tKeyBoard Operation\n\n"
		"\tMouse click Image to floodFill\n"
		"\tKeyBoard[q] - Return Main Meun\n"
		"\tKeyBoard[1] - RGB/Gray\n"
		"\tKeyBoard[2] - show/hide Mask Window\n"
		"\tKeyBoard[3] - Refresh\n"
		"\tKeyBoard[4] - floodFill type1\n"
		"\tKeyBoard[5] - floodFill type2\n"
		"\tKeyBoard[6] - floodFill type3\n"
		"\tKeyBoard[7] - 4bit connect mode\n"
		"\tKeyBoard[8] - 8bit connect mode\n\n"
		);

	g_srcImage = imread("lena.jpg");
	imshow("Source Image", g_srcImage);

	g_srcImage.copyTo(g_dstImage);
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	g_maskImage.create(g_srcImage.rows + 2, g_srcImage.cols + 2, CV_8UC1);

	namedWindow(WINDOW_NAME_50);
	createTrackbar("minus_max:", WINDOW_NAME_50, &g_nLowDifference, 255, 0);
	createTrackbar("plus_max:", WINDOW_NAME_50, &g_nUpDifference, 255, 0);
	setMouseCallback(WINDOW_NAME_50, on_MouseHandle_P50, 0);

	while (1)
	{
		imshow(WINDOW_NAME_50, g_bIsColor ? g_dstImage : g_grayImage);

		int c = waitKey(0);
		if (c == (int)'q')
		{
			cout << "Retrun back Main Meun.";
			system("cls");
			system("color 3F");
			ShowHelpText();
			destroyAllWindows();
			P0_LoadSourceImage();
			break;
		}
		switch ((char)c)
		{
		case '1':
			if (g_bIsColor)
			{
				cout << "[1]Image Color Model Change:RGB-->Gray" << endl;
				cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
				g_maskImage = Scalar::all(0);
				g_bIsColor = false;
			}
			else
			{
				cout << "[1]Image Color Model Change:Gray-->RGB" << endl;
				g_srcImage.copyTo(g_dstImage);
				g_maskImage = Scalar::all(0);
				g_bIsColor = true;
			}
			break;
		case '2':
			if (g_bUserMask)
			{
				cout << "[2]Mask Window Change:Hide" << endl;
				destroyWindow("Mask");
				g_bUserMask = false;
			}
			else
			{
				cout << "[2]Mask Window Change:Show" << endl;
				namedWindow("Mask", 0);
				g_maskImage = Scalar::all(0);
				imshow("Mask", g_maskImage);
				g_bUserMask = true;
			}
			break;
		case '3':
			cout << "[3]Refresh Source Image" << endl;
			g_srcImage.copyTo(g_dstImage);
			cvtColor(g_dstImage, g_grayImage, COLOR_BGR2GRAY);
			g_maskImage = Scalar::all(0);
			break;
		case '4':
			cout << "[4]floodFill type1" << endl;
			g_FillMode = 0;
			break;
		case '5':
			cout << "[5]floodFill type2" << endl;
			g_FillMode = 1;
			break;
		case '6':
			cout << "[6]floodFill type3" << endl;
			g_FillMode = 2;
			break;
		case '7':
			cout << "[7]4bit connect mode" << endl;
			g_nConnectivity = 4;
			break;
		case '8':
			cout << "[8]8bit connect mode" << endl;
			g_nConnectivity = 8;
			break;

		}
	}

}
void on_MouseHandle_P50(int event, int x, int y, int, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	Point seed = Point(x, y);
	int LowDifference = g_FillMode == 0 ? 0 : g_nLowDifference;
	int UpDifference = g_FillMode == 0 ? 0 : g_nUpDifference;
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) + (g_FillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);

	int b = (unsigned)theRNG() % 255;
	int g = (unsigned)theRNG() % 255;
	int r = (unsigned)theRNG() % 255;
	Rect ccomp;

	Scalar newVel = g_bIsColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;
	int area;

	if (g_bUserMask)
	{
		threshold(g_maskImage, g_maskImage, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dst, g_maskImage, seed, newVel, &ccomp, Scalar(LowDifference, LowDifference, LowDifference), Scalar(UpDifference, UpDifference, UpDifference), flags);
		imshow("Mask", g_maskImage);
	}
	else
	{
		area = floodFill(dst, seed, newVel, &ccomp, Scalar(LowDifference, LowDifference, LowDifference), Scalar(UpDifference, UpDifference, UpDifference), flags);
	}
	imshow(WINDOW_NAME_50, dst);
	cout << "Number of repaint pixel: "<< area << endl;
}
void P51_Resize()
{
	g_srcImage = imread("lena.jpg");
	
	resize(g_srcImage, g_dstImage1, Size(g_srcImage.cols/2,g_srcImage.rows/2), 0.0, 0.0, 3);
	resize(g_srcImage, g_dstImage2, Size(g_srcImage.cols*2,g_srcImage.rows*2), 0.0, 0.0, 3);
	namedWindow(WINDOW_NAME_51_1);
	imshow(WINDOW_NAME_51_1, g_dstImage1);
	namedWindow(WINDOW_NAME_51_2);
	imshow(WINDOW_NAME_51_2, g_dstImage2);

}
void P54_PyrAndResize()
{
	destroyAllWindows();
	system("cls");
	system("color 2F");
	printf("\n\n\n\tWelcome to Pyr and Resize\n\n");
	printf("\n\n\tKeyBoard Operation\n\n"
		"\tMouse click Image to floodFill\n"
		"\tKeyBoard[q] - Return Main Meun\n"
		"\tKeyBoard[a] - resize *2\n"
		"\tKeyBoard[d] - resize /2\n"
		"\tKeyBoard[w] - PyrUp *2\n"
		"\tKeyBoard[s] - PyrDown /2\n\n"
	);

	g_srcImage = imread("lena.jpg");
	namedWindow(WINDOW_NAME_54);
	imshow(WINDOW_NAME_54, g_srcImage);

	g_tmpImage = g_srcImage;
	g_dstImage = g_tmpImage;

	int key = 0;
	while (1)
	{
		key = waitKey(9);
		switch (key)
		{
		case 'q':
			cout << "Retrun back Main Meun.";
			system("cls");
			system("color 3F");
			ShowHelpText();
			destroyAllWindows();
			P0_LoadSourceImage();
			return;
		case 'w':
			cout << "[w]pyrUp->Size*2" << endl;
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			break;
		case 's':
			cout << "[s]pyrDown->Size/2" << endl;
			pyrDown(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			break;
		case 'a':
			cout << "[a]resize->Size*2" << endl;
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			break;
		case 'd':
			cout << "[d]resize->Size/2" << endl;
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			break;
		}
		imshow(WINDOW_NAME_54, g_dstImage);
		g_tmpImage = g_dstImage;
	}
}
void P55_Threshold()
{
	g_srcImage = imread("lena.jpg");
	namedWindow(WINDOW_NAME_55);

	createTrackbar("Type:", WINDOW_NAME_55, &g_nThresholdTypeValue, 4, on_Threshold);
	on_Threshold(g_nThresholdTypeValue, 0);
	createTrackbar("Value:", WINDOW_NAME_55, &g_nThresholdValue, 255, on_Threshold);
	on_Threshold(g_nThresholdValue, 0);

}
void on_Threshold(int, void*)
{
	cvtColor(g_srcImage, g_tmpImage, COLOR_BGR2GRAY);
	threshold(g_tmpImage, g_dstImage, g_nThresholdValue, 255, g_nThresholdTypeValue);

	imshow(WINDOW_NAME_55, g_dstImage);
}
void P56_Canny()
{
	g_srcImage = imread("lena.jpg");
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	//simple
	g_dstImage1.create(g_srcImage.size(), g_srcImage.type());
	Canny(g_srcImage, g_dstImage1, 150, 90);
	imshow("simple", g_dstImage1);

	//high level
	blur(g_grayImage, g_tmpImage, Size(3, 3));
	Canny(g_tmpImage, g_tmpImage, 3, 9);
	g_dstImage = Scalar::all(0);
	g_srcImage.copyTo(g_dstImage, g_tmpImage);

	namedWindow(WINDOW_NAME_56);
	imshow(WINDOW_NAME_56, g_dstImage);
	
}
void P57_Sobel()
{
	Mat abs_grad_x, abs_grad_y,abs_dstImage;
	g_srcImage = imread("lena.jpg");

	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

	Sobel(g_grayImage, g_gradImage_x, g_grayImage.depth(), 1, 0);
	convertScaleAbs(g_gradImage_x, abs_grad_x);
	namedWindow(WINDOW_NAME_57_x);
	imshow(WINDOW_NAME_57_x, abs_grad_x);

	Sobel(g_grayImage, g_gradImage_y, g_grayImage.depth(), 0, 1);
	convertScaleAbs(g_gradImage_y, abs_grad_y);
	namedWindow(WINDOW_NAME_57_y);
	imshow(WINDOW_NAME_57_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, abs_dstImage);
	namedWindow(WINDOW_NAME_57);
	imshow(WINDOW_NAME_57, abs_dstImage);

	//WritePixelToFile(abs_dstImage, "Sobel.txt");

}
void P58_Laplacian()
{
	Mat abs_dstImage;
	g_srcImage = imread("lena.jpg");
	GaussianBlur(g_srcImage, g_tmpImage, Size(3, 3), 0, 0);
	cvtColor(g_tmpImage, g_grayImage, COLOR_BGR2GRAY);

	Laplacian(g_grayImage, g_dstImage, g_grayImage.depth(),3,1,0,BORDER_DEFAULT);
	convertScaleAbs(g_dstImage, abs_dstImage);
	
	namedWindow(WINDOW_NAME_58);
	imshow(WINDOW_NAME_58, abs_dstImage);

	//WritePixelToFile(abs_dstImage, "Laplacian.txt");

}
void P59_Scharr()
{
	Mat abs_grad_x, abs_grad_y;
	g_srcImage = imread("lena.jpg");

	Scharr(g_srcImage, g_gradImage_x, g_srcImage.depth(), 1, 0);
	convertScaleAbs(g_gradImage_x, abs_grad_x);
	namedWindow(WINDOW_NAME_59_x);
	imshow(WINDOW_NAME_59_x, abs_grad_x);

	Scharr(g_srcImage, g_gradImage_y, g_srcImage.depth(), 0, 1);
	convertScaleAbs(g_gradImage_y, abs_grad_y);
	namedWindow(WINDOW_NAME_59_y);
	imshow(WINDOW_NAME_59_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, g_dstImage);
	namedWindow(WINDOW_NAME_59);
	imshow(WINDOW_NAME_59, g_dstImage);

}
void P60_EdgeDetection()
{
	g_srcImage = imread("lena.jpg");
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

	namedWindow(WINDOW_NAME_60_1);
	createTrackbar("Low:", WINDOW_NAME_60_1, &g_nCannyLowThresholdValue, 120, on_Canny);
	on_Canny(g_nCannyLowThresholdValue, 0);
	createTrackbar("High:", WINDOW_NAME_60_1, &g_nCannyHighThresholdValue, 255, on_Canny);
	on_Canny(g_nCannyHighThresholdValue, 0);

	namedWindow(WINDOW_NAME_60_2);
	createTrackbar("Kernel:", WINDOW_NAME_60_2, &g_nSobelKernelSizeValue, 3, on_Sobel);
	on_Sobel(g_nSobelKernelSizeValue, 0);
}
void on_Canny(int, void*)
{
	Mat CannySrcImage;
	if (g_nCannyHighThresholdValue <= g_nCannyLowThresholdValue)g_nCannyHighThresholdValue = g_nCannyLowThresholdValue * 3;

	//blur(g_grayImage, CannySrcImage, Size(3, 3));
	GaussianBlur(g_grayImage, CannySrcImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Canny(CannySrcImage, CannySrcImage, g_nCannyLowThresholdValue, g_nCannyHighThresholdValue, 3);
	g_dstImage = Scalar::all(0);
	g_srcImage.copyTo(g_dstImage, CannySrcImage);

	imshow(WINDOW_NAME_60_1, g_dstImage);

}
void on_Sobel(int, void*)
{
	Mat SobelImage_x, SobelImage_y, abs_SobelImage_x, abs_SobelImage_y;
	Sobel(g_srcImage, SobelImage_x, g_srcImage.depth(), 1, 0,g_nSobelKernelSizeValue*2+1);
	convertScaleAbs(SobelImage_x, abs_SobelImage_x);

	Sobel(g_srcImage, SobelImage_y, g_srcImage.depth(), 0,1, g_nSobelKernelSizeValue * 2 + 1);
	convertScaleAbs(SobelImage_y, abs_SobelImage_y);

	addWeighted(abs_SobelImage_x, 0.5, abs_SobelImage_y, 0.5, 0, g_dstImage);

	imshow(WINDOW_NAME_60_2, g_dstImage);
}
void P61_HoughLines()
{
	g_srcImage = imread("lena.jpg");
	
	Canny(g_srcImage, g_tmpImage, 50, 200, 3);
	cvtColor(g_tmpImage, g_dstImage, COLOR_BayerGB2GRAY);

	g_dstImage1.create(g_dstImage.size(),g_dstImage.type());
	g_dstImage1 = Scalar::all(0);

	vector<Vec2f>lines;
	HoughLines(g_tmpImage, lines, 1, CV_PI / 180, 150, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(g_dstImage, pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
		line(g_dstImage1, pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
	}

	imshow("Canny TmpImage", g_tmpImage);
	imshow("Line Image", g_dstImage1);
	namedWindow(WINDOW_NAME_61);
	imshow(WINDOW_NAME_61, g_dstImage);
}

#pragma endregion


void Test()
{
	//boxFilter;
	//RNG& rng = theRNG();
	////srand(time(0));
	//for (int i = 0; i < 100; i++)
	//{
	//	cout<< rng.uniform(0, 99) <<endl;
	//	//cout << rand() << endl;
	//}
	
	g_dstImage = imread("lena.jpg",0);
	WritePixelToFile(g_dstImage, "1.txt");
}
