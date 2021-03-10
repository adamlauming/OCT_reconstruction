/*==============================================================================================
Program: reconstruction.h
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/11/25
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#pragma once
#ifndef BV_RECONSTRUCTION_RECONSTRUCTION_H
#define BV_RECONSTRUCTION_RECONSTRUCTION_H

#include <opencv2/opencv.hpp>

class Mat_Data;

class FFT
{
public:
    FFT();
    ~FFT();
    void SetPara(double* para);
    void SetPara(double p0, double p1, double p2, double p3) {
        double para[4] = { p0, p1, p2, p3 };
        SetPara(para);
    };
    void GetOCTDFTImg_CPU(const cv::Mat& input, cv::Mat& output, bool SUB_BKG = true);

protected:
    /**
     * @brief:   1D data DFT over column image by CPU
     * @details: Use interfaces in OpenCV. Do DFT over each col.
     * @param:   const cv::Mat & input, column data and the size is 2048 * 1
     * @param:   cv::Mat & output, the size is the same with input
     * @note: No quadrants exchange for speeding up algorithms.
    */
    void DFT1D_CPU(const cv::Mat& input, cv::Mat& output);

private:
    /**
     * @brief:   Initial wavelength and map it to wave-number space
    */
    void InitWave();
    double c0, c1, c2, c3;
    double *_kCamera, *_kResort, *_interFactor;
    int *_interIndex;
    std::vector<cv::Mat> phase_channels;
    Mat_Data * matdata;
};

#endif //BV_RECONSTRUCTION_RECONSTRUCTION_H
