// Shallow_Detection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include<gdal.h>
#include<gdal_priv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp> 
#include<vector>
using namespace cv;
using namespace std;

////////////img转Mat////////////
Mat GDal2OpenCV(GDALDataset* dataset, int BandNum) {
    // 获取图像的宽度、高度和波段数  
    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    int bands = dataset->GetRasterCount();

    GDALRasterBand* band = dataset->GetRasterBand(BandNum);
    GDALDataType dataType = band->GetRasterDataType();

    // 根据GDAL数据类型选择OpenCV类型  
    int cvType;
    switch (dataType) {
    case GDT_Byte: cvType = CV_8UC1; break;
    case GDT_UInt16: cvType = CV_16UC1; break;
    case GDT_Int16: cvType = CV_16SC1; break;
    case GDT_Int32: cvType = CV_32SC1; break;
    case GDT_Float32: cvType = CV_32FC1; break;
    case GDT_Float64: cvType = CV_64FC1; break;
    default:
        cerr << "Unsupported GDAL data type." << endl;
        return Mat();
    }
    // 为OpenCV Mat分配内存  
    Mat mat(height, width, cvType);
    band->RasterIO(GF_Read, 0, 0, width, height, mat.data, width, height, dataType, 0, 0);

    return mat;
}

////////////均衡化////////////
Mat GLT_Colored(Mat mat)
{
    Mat glt_result;
    vector<Mat> chans;
    split(mat, chans);
    for (auto& chan : chans) {//分通道进行拉伸
        vector<uchar> pix_val;//存放各个像素的值
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                pix_val.push_back(chan.at<uchar>(i, j));
            }
        }
        sort(pix_val.begin(), pix_val.end());//排序以确定分位数
        double min_val = pix_val[0.02 * pix_val.size()];
        double max_val = pix_val[0.98 * pix_val.size()];
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                chan.at<uchar>(i, j) = max(0.0, min(255.0, (chan.at<uchar>(i, j) - min_val) / (max_val - min_val) * 255));
            }
        }
    }
    merge(chans, glt_result);
    //glt_result.convertTo(glt_result, CV_8U, 256.0 / 65536.0);
    return glt_result;
}

////////////HSV////////////
void Shadow_detection_HSV(Mat img)
{
    int width = img.cols;
    int height = img.rows;
    Mat dst,temp;
    dst.create(height,width,CV_8U);
    temp.create(height, width, CV_8UC3);
    cvtColor(img, temp, COLOR_RGB2HSV);
    unsigned char* p1 = dst.data;
    unsigned char* p = temp.data;
    double h, s, v, m;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            h = p[(i * width + j) * 3 + 0];
            s = p[(i * width + j) * 3 + 1];
            v = p[(i * width + j) * 3 + 2];
            m = double(((s - v) / (h + s + v)));
            if (m > 0)
            {
                p1[i * width + j] = 255;
            }
            else
            {
                p1[i * width + j] = 0;
            }
        }
    }

    // 定义结构元素（例如，3x3的矩形）  
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // 进行闭运算  
    Mat result;
    morphologyEx(dst, result, MORPH_CLOSE, kernel);
    
    imwrite(".\\Shadow-hsv_close.png", result);
}

////////////C1C2C3////////////
void Shadow_detection_C1C2C3(Mat img)
{
    int row = img.rows, col = img.cols;
    Mat dst(row, col, CV_8UC1);
    double temp;
    double r, g, b, c1, c2, c3, gb_max, rb_max, rg_max;

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            b = (double)img.at<Vec3b>(i, j)[0];
            g = (double)img.at<Vec3b>(i, j)[1];
            r = (double)img.at<Vec3b>(i, j)[2];
            if (b > g) gb_max = b;
            else gb_max = g;
            if (b > r) rb_max = b;
            else rb_max = r;
            if (r > g) rg_max = r;
            else rg_max = g;
            c1 = atan(r / gb_max);
            c2 = atan(g / rb_max);
            c3 = atan(b / rg_max);
            temp = c3;
            if (temp > 0.4 && img.at<Vec3b>(i, j)[0] < 110) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }


    Mat kernel = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
    // 进行闭运算  
    Mat result;
    morphologyEx(dst, result, MORPH_CLOSE, kernel);
    imwrite(".\\Shadow-c1c2c3_close.png", result);
}


int main()
{
    //使用gdal读取img影像数据
    GDALAllRegister();//注册gdal驱动程序

    // 打开img
    const char* pszFilename = "E:/part3/zy-3-wd.img";
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(pszFilename, GA_ReadOnly);
    if (poDataset == NULL)
    {
        cout << "Failed to open image file: " << pszFilename << endl;
        exit(1);
    }

    // Get image dimensions and bands
    int nXSize = poDataset->GetRasterXSize();
    int nYSize = poDataset->GetRasterYSize();
    int nBands = poDataset->GetRasterCount();
    Mat* bands = new Mat[nBands + 1];//索引0设置为空
    Mat temp;
    for (int i = 1; i <= nBands; i++) {//读取资源卫星1-4波段波段影像
        temp = GDal2OpenCV(poDataset, i);
        temp.convertTo(bands[i], CV_8UC1,1.0/2.0);//转换为8位无符号型
    }
    GDALClose(poDataset);
    
    Mat rgb_image;

    // 合并波段到RGB图像中
    vector<Mat> channels;
    channels.push_back(bands[1]);
    channels.push_back(bands[2]);
    channels.push_back(bands[3]);
    merge(channels, rgb_image);


    //rgb_image = GLT_Colored(rgb_image);

    // 保存RGB图像  
    //imwrite("E:/part3/rgb_image_GLT .png", rgb_image);

    // 显示RGB图像  
    //imshow("RGB Image", rgb_image);
    //waitKey(0);

    Shadow_detection_HSV(rgb_image);
    Shadow_detection_C1C2C3(rgb_image);
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
