/*==============================================================================================
Program: other.cpp
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/11/26
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include"other.h"

Mat_Data* Mat_Data::_instance;
Mat_Data::Mat_Data() {
    wave_to_pix[0] = 8.00608E+02;  wave_to_pix[1] = 4.13653E-02;  wave_to_pix[2] = -1.23689E-06;  wave_to_pix[3] = -5.40202E-11;
}
Mat_Data* Mat_Data::GetInstance() {     //单例
    if (!_instance) {
        _instance = new Mat_Data();
    }
    return _instance;
}
Mat_Data::~Mat_Data(void) {
    delete[] phase_para;
}