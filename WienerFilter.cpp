/*==============================================================================================
Program: WienerFilter.cpp
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/12/13
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include "WienerFilter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace cv;

double WienerFilterImpl(const Mat& src, Mat& dst, double noiseVariance, const Size& block){

//    assert(("Invalid block dimensions", block.width % 2 == 1 && block.height % 2 == 1 && block.width > 1 && block.height > 1));
//    assert(("src and dst must be one channel grayscale images", src.channels() == 1, dst.channels() == 1));

    int h = src.rows;
    int w = src.cols;

    dst = Mat1b(h, w);

    Mat1d means, sqrMeans, variances;
    Mat1d avgVarianceMat;

    boxFilter(src, means, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);
    sqrBoxFilter(src, sqrMeans, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);

    Mat1d means2 = means.mul(means);
    variances = sqrMeans - (means.mul(means));

    if (noiseVariance < 0){
        // I have to estimate the noiseVariance
        reduce(variances, avgVarianceMat, 1, REDUCE_SUM, -1);
        reduce(avgVarianceMat, avgVarianceMat, 0, REDUCE_SUM, -1);
        noiseVariance = avgVarianceMat(0, 0) / (h*w);
    }

    for (int r = 0; r < h; ++r){
        // get row pointers
        uchar const * const srcRow = src.ptr<uchar>(r);
        uchar * const dstRow = dst.ptr<uchar>(r);
        double * const varRow = variances.ptr<double>(r);
        double * const meanRow = means.ptr<double>(r);
        for (int c = 0; c < w; ++c) {
            dstRow[c] = saturate_cast<uchar>(
                    meanRow[c] + max(0., varRow[c] - noiseVariance) / max(varRow[c], noiseVariance) * (srcRow[c] - meanRow[c])
            );
        }
    }

    return noiseVariance;
}

void WienerFilter(const Mat& src, Mat& dst, double noiseVariance, const Size& block){
    WienerFilterImpl(src, dst, noiseVariance, block);
    return;
}


double WienerFilter(const Mat& src, Mat& dst, const Size& block){
    return WienerFilterImpl(src, dst, -1, block);
}


