/*==============================================================================================
Program: other.h
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/11/25
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#pragma once
#ifndef OTHER_H
#define OTHER_H
#include <opencv2/opencv.hpp>
class FFT;

class Mat_Data {
public:
    static Mat_Data* GetInstance();
    std::vector<cv::Mat> Img;
    int slice = 1;
    int height = 950; //950 //1000 //1024
    double *wave_to_pix = new double[4];
    double * phase_para = new double[2048];
    bool compensaion_state = false;
    bool Img_Process_State = true;
    float Img_Threshold = 0.6;
private:
    Mat_Data();
    virtual ~Mat_Data();
    static Mat_Data* _instance;
};

#endif //BV_RECONSTRUCTION_OTHER_H
