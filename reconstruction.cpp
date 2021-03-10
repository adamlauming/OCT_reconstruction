/*==============================================================================================
Program: reconstruction.cpp
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/11/25
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include "reconstruction.h"
#include "other.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;
const int ASCAN_LEN = 2048;
const double My_PI = 3.14159265358979323846;
FFT::FFT() : c0(0.0), c1(0.0), c2(0.0), c3(0.0),
        _kCamera(new double[ASCAN_LEN]),
        _kResort(new double[ASCAN_LEN]),
        _interIndex(new int[ASCAN_LEN]),
        _interFactor(new double[ASCAN_LEN]) {
    memset(_interIndex, 0, sizeof(int) * ASCAN_LEN);
    memset(_interFactor, 0.0, sizeof(double) * ASCAN_LEN);
    phase_channels.resize(2);
    phase_channels[0].create(1, 2048, CV_32FC1);//相位参数实部
    phase_channels[1].create(1, 2048, CV_32FC1);
}

//----------------------------------------------------------------------------
FFT::~FFT() {
    if (_interIndex) delete[] _interIndex;
    if (_interFactor) delete[] _interFactor;
    if (_kResort) delete[] _kResort;
    if (_kCamera) delete[] _kCamera;
}

//----------------------------------------------------------------------------
void FFT::SetPara(double* para) {
    if (!para) return;
    c0 = para[0];
    c1 = para[1];
    c2 = para[2];
    c3 = para[3];
    InitWave();
}

void FFT::InitWave() {
    // [1] Get corresponding wavelength array
    double lamb[ASCAN_LEN] = { 0.0 };	//wavelength array
#pragma omp parallel for
    for (int i = 0; i < ASCAN_LEN; ++i)
        lamb[i] = c0 + c1 * i + c2 * i * i + c3 * i * i * i;	// faster than pow method

    // [2] Mapping the wavelength into the wave-number space
#pragma omp parallel for
    for (int i = 0; i < ASCAN_LEN; ++i)
        _kCamera[i] = 2000 * My_PI / lamb[i];

    // [3] Interpolation to make the KCamera array has an equal interval
    double interval = (_kCamera[ASCAN_LEN - 1] - _kCamera[0]) / (ASCAN_LEN - 1);
#pragma omp parallel for
    for (int i = 0; i < ASCAN_LEN; ++i) {
        _kResort[i] = _kCamera[0] + i * interval;
    }

    //[4] Linear resampling
#pragma omp parallel for
    for (int i = 0; i < ASCAN_LEN; ++i) {
#pragma omp parallel for
        for (int j = 0; j < ASCAN_LEN - 1; j++) {
            if (_kResort[i] <= _kCamera[j] && _kResort[i] >= _kCamera[j + 1]) {
                _interIndex[i] = j;
                _interFactor[i] = (_kResort[i] - _kCamera[j]) / (_kCamera[j + 1] - _kCamera[j]);
            }
        }
    }

    matdata = Mat_Data::GetInstance();
    for (int i = 0; i < 2048; i++) {
        float * p_real = phase_channels[0].ptr<float>(0);
        float * p_imag = phase_channels[1].ptr<float>(0);
        p_real[i] = cos(matdata->phase_para[i]);
        p_imag[i] = sin(matdata->phase_para[i]);
    }
}

//----------------------------------------------------------------------------
void FFT::GetOCTDFTImg_CPU(const cv::Mat& input, cv::Mat& output, bool SUB_BKG /*= true*/) {
    //output.create(input.size(), CV_32FC1);
    if (input.empty()) {
        printf("Empty Image\n");
        return;
    }
    Mat tempData = Mat(input.size(), CV_32FC1);
#pragma omp parallel for
    for (int i = 0; i < input.rows; ++i) {
#pragma omp parallel for
        for (int j = 0; j < ASCAN_LEN; ++j) {
            tempData.at<float>(i, j) = input.at<unsigned short>(i, _interIndex[j]) + (_interFactor[j] *
                                                                                      ((float)input.at<unsigned short>(i, _interIndex[j] + 1) - input.at<unsigned short>(i, _interIndex[j])));
        }
    }

    // Subtract the Background Noise
    if (SUB_BKG) {
        double ave[ASCAN_LEN];
        double bkg[ASCAN_LEN];
#pragma omp parallel for
        for (int i = 0; i < ASCAN_LEN; ++i)
            ave[i] = mean(input.col(i))[0];

#pragma omp parallel for
        for (int i = 0; i < ASCAN_LEN; i++)
            bkg[i] = ave[_interIndex[i]] + _interFactor[i] * (ave[_interIndex[i] + 1] - ave[_interIndex[i]]);

#pragma omp parallel for
        for (int i = 0; i < tempData.rows; ++i) {
            float *p = tempData.ptr<float>(i);
            for (int j = 0; j < tempData.cols; ++j) {
                p[j] = p[j] - bkg[j];
            }
        }
    }

    DFT1D_CPU(tempData, output);

    //#pragma omp parallel for
    //	for (int j = 0; j < input.cols; ++j)
    //		DFT1D_CPU(tempData.col(j), output.col(j));

    //Mat tmp;
    //Mat q1 = output(Rect(0, 0, output.cols, output.rows / 2));
    //Mat q2 = output(Rect(0, output.rows / 2, output.cols, output.rows / 2));
    //q1.copyTo(tmp);
    //q2.copyTo(q1);
    //tmp.copyTo(q2);
}

//----------------------------------------------------------------------------
void FFT::DFT1D_CPU(const cv::Mat& input, cv::Mat& output) {
    matdata = Mat_Data::GetInstance();
    //output.create(input.size(), CV_32FC1);
    if (input.empty() || input.cols != ASCAN_LEN) {
        printf("Input data must have %d cols\n", ASCAN_LEN);
        return;
    }
    // [2] Get the optimal dft size and initialize pixels in the padding to zero
    //int m = getOptimalDFTSize(input.rows);
    //int n = getOptimalDFTSize(input.cols);
    Mat padded;
    // Must be implemented, otherwise the time cost will be much higher.
    if (matdata->compensaion_state) {
        std::vector <cv::Mat> channels;
        channels.resize(2);
        channels[0].create(input.size(), CV_32FC1);
        channels[1].create(input.size(), CV_32FC1);

        copyMakeBorder(input, channels[0], 0, 0, 0, 0, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(input, channels[1], 0, 0, 0, 0, BORDER_CONSTANT, Scalar::all(0));

        for (int i = 0; i < input.rows; i++) {
            channels[0].row(i) = channels[0].row(i).mul(phase_channels[0]);
            channels[1].row(i) = channels[1].row(i).mul(phase_channels[1]);
        }

        merge(channels, padded);
    }
    else {
        copyMakeBorder(input, padded,
                       0, 0, 0, 0,
                       BORDER_CONSTANT, Scalar::all(0));
    }

    Mat complexI;
    dft(padded, complexI, DFT_COMPLEX_OUTPUT | DFT_ROWS);
    Mat planes[] = { Mat::zeros(complexI.size(), CV_32FC1), Mat::zeros(complexI.size(), CV_32FC1) };
    split(complexI, planes);
    Mat mag;
    magnitude(planes[0], planes[1], mag);
    mag += Scalar(1);
    log(mag, mag);
    Mat roi(mag, Rect(mag.cols / 2, 0, mag.cols / 2, mag.rows));
    transpose(roi, output);

    if (matdata->Img_Process_State) {
//        double OCT_Max, OCT_Min;
//        minMaxLoc(output, NULL, &OCT_Max, NULL, NULL);
//        output = output / OCT_Max;
        Scalar tempVal = mean(output);
        double matMean = tempVal.val[0];
//        cout << matMean << endl;
        double OCT_Max, OCT_Min;
        minMaxLoc(output, NULL, &OCT_Max, NULL, NULL);
//        cout << OCT_Max << endl;
        output = output / OCT_Max;
        matdata->Img_Threshold = 1.1 * matMean / OCT_Max ; /// 1.1 is adjusted.

        for (int i = 0; i < output.rows; i++) {
            float * p = output.ptr<float>(i);
            for (int j = 0; j < output.cols; j++) {

                if (p[j] < matdata->Img_Threshold) {
                    p[j] = matdata->Img_Threshold;
                }
                p[j] = 20 * std::log10f(p[j]);
            }
        }
        minMaxLoc(output, &OCT_Min, NULL, NULL, NULL);
        output = output - OCT_Min;
    }



    //// [3] Allocate storage for the result of DFT
    //// combine arrays of planes into complexI which has multiple channels
    //Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    //Mat complexI;
    //merge(planes, 2, complexI);

    //// [4] DFT in-place
    //dft(complexI, complexI);

    //// [5] Translate the complex into the magnitude
    //// split complexI into several arrays of single channel
    //// planes[0] = Re, planes[1] = im
    //split(complexI, planes); //complexI
    //magnitude(planes[0], planes[1], planes[0]);	//planes[0] = magnitude
    //Mat magnitudeImg = planes[0];

    //// [6] logarithmic scale
    //magnitudeImg += Scalar(1);
    //log(magnitudeImg, magnitudeImg);
    //magnitudeImg.copyTo(output);
}



