/*==============================================================================================
Program: WienerFilter.h
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/12/13
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <assert.h>

/** @brief Implementation of the adaptive Wiener filter

This function applies to the src image the adaptive Wiener filter and
store the result in the dst image. The formula that will be apply is
the following one:

dst(x, y) = u + max(0, s^2 - v^2)(src(x, y) - u) / max(s^2, v^2)

where u is the local mean, s^2 is the variance at NxM neighborhood
around each pixel (they depend on block dimension) e v^2 is the noise
variance calculated as the average of all the local estimated variances
if not given.

@param[in] src input grayscale image (Mat1b)
@param[out] dst output grayscale image (Mat1b)
@param[in] block dimension of the block (width, height) to use in order
			to compute the filtering process, default is 5x5

@return estimated noise variance
*/
double WienerFilter(const cv::Mat& src, cv::Mat& dst, const cv::Size& block = cv::Size(5, 5));

/** @overload

@param[in] src input grayscale image (Mat1b)
@param[out] dst output grayscale image (Mat1b)
@param[in] noiseVariance noise variance to use in order to calculate Wiener filter (must be positive)
@param[in] block dimension of the block (width, height) to use in order
			to compute the filtering process, default is 5x5

@return estimated noise variance
*/
void WienerFilter(const cv::Mat& src, cv::Mat& dst, double noiseVariance, const cv::Size& block = cv::Size(5, 5));