/*==============================================================================================
Program: main.cpp
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2019/11/28
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2019-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include "reconstruction.h"
#include "WienerFilter.h"
#include "other.h"
#include "SGSmooth.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <math.h>
#include <numeric>

using namespace cv;
using namespace std;

// Sort vector and return the index, please notice that the vector is unchangable. //vec中的内容不变,返回排序后的索引值
template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}

// Mat to vector
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
    return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}

// Calculate the cross_correlation coefficient.
double mat_corr(Mat& mat1, Mat& mat2) {
    if ((mat1.rows != mat2.rows)||(mat1.cols != mat2.cols)){
        cout << "please input 2 images of the same size" << endl;
        return -1;
    }

    else{
        double sumA(0.0), sumB(0.0), aveA(0.0), aveB(0.0);
        long Length = mat1.rows * mat1.cols;

        vector<double> A = convertMat2Vector<double>(mat1);
        vector<double> B = convertMat2Vector<double>(mat2);

        //求和
        sumA = std::accumulate(A.begin(), A.end(), 0.0);
        sumB = std::accumulate(B.begin(), B.end(), 0.0);

        //求平均值
        aveA = sumA / double(Length);
        aveB = sumB / double(Length);

        //计算相关系数
        double R1(0), R2(0), R3(0);
        for (long i = 0; i < Length; i++)
        {
            R1 += (A[i] - aveA) * (B[i] - aveB);
            R2 += pow((A[i] - aveA), 2);
            R3 += pow((B[i] - aveB), 2);
        }

        return (R1 / sqrt(R2*R3));
    }
}

// CircShift
inline void circRowShift(Mat& src,int shift_m_rows)
{
    int m=shift_m_rows;
    int rows=src.rows;
    //‘行’循环移动
    if(m%rows==0)
    {
        return;
    }

    Mat mrows(abs(m),src.cols,src.type());//用于暂时保存末尾的m行数据
    Mat temp = Mat::zeros(src.size(), CV_8UC1);
    if(m>0)
    {
        src(Range(rows-m,rows),Range::all()).copyTo(mrows);
        src(Range(0,rows-m),Range::all()).copyTo(temp(Range(m,rows),Range::all()));
        mrows.copyTo(temp(Range(0,m),Range::all()));
        temp.copyTo(src);
    }else
    {
        src(Range(0,-m),Range::all()).copyTo(mrows);
        src(Range(-m,rows),Range::all()).copyTo(temp(Range(0,rows+m),Range::all()));
        mrows.copyTo(temp(Range(rows+m,rows),Range::all()));
        temp.copyTo(src);
    }
}
inline void circColShift(Mat& src,int shift_n_cols)
{
    int n=shift_n_cols;
    int cols=src.cols;
    int rows=src.rows;
    if(n%cols==0)
    {
        return;
    }
    ///ncols，如果n>0，用于暂时保存末尾的n列数据
    ///ncols，如果n<0，用于暂时保存起始的n列数据
    Mat ncols(rows,abs(n),src.type());
    Mat temp = Mat::zeros(src.size(), CV_8UC1);
    if(n>0)
    {
        src(Range::all(),Range(cols-n,cols)).copyTo(ncols);
        src(Range::all(),Range(0,cols-n)).copyTo(temp(Range::all(),Range(n,cols)));
        ncols.copyTo(temp(Range::all(),Range(0,n)));
        temp.copyTo(src);
    }
    else
    {
        src(Range::all(),Range(0,-n)).copyTo(ncols);
        src(Range::all(),Range(-n,cols)).copyTo(temp(Range::all(),Range(0,cols+n)));
        ncols.copyTo(temp(Range::all(),Range(cols+n,cols)));
        temp.copyTo(src);
    }
}
void circShift(Mat& inImg, Mat& outImg, int shift_m_rows,int shift_n_cols)
{
    outImg = inImg.clone();
    int m=shift_m_rows;
    int n=shift_n_cols;
    //‘行’循环移动
    circRowShift(outImg,m);
    //‘列’循环移动
    circColShift(outImg,n);
}

// Find the base image and the most similar images for registration and the number is 80% of the total images.
void FindBaseImage(const std::vector<cv::Mat>& vecInImgs, cv::Mat& outImg, std::vector<cv::Mat>& vecoutImgs){
    // Average the raw images, find the most similar image to aveImg and this image is the base image.
    vector<double> array_temp;
    Mat aveImg;
    aveImg = Mat::zeros(vecInImgs[0].size(),CV_32FC1);
    for (auto& img : vecInImgs) {
        Mat dst;
        img.convertTo(dst, CV_32FC1);
        aveImg += dst;
    }
    aveImg /= vecInImgs.size();
    aveImg.convertTo(aveImg, CV_8UC1);

    for (int k = 0; k < vecInImgs.size(); k++){
        double temp;
        Mat imgtemp = vecInImgs[k];
        temp = mat_corr(aveImg,imgtemp);
        array_temp.push_back(temp);
    }

    std::vector<double>::iterator biggest = std::max_element(std::begin(array_temp), std::end(array_temp));
    int BasePosition = std::distance(std::begin(array_temp), biggest);
    cout << "The base image is at position " << BasePosition << endl;
    vecInImgs[BasePosition].convertTo(outImg, CV_8UC1);
    imwrite("image_base.png",outImg);

    // look for the images that are most similar to the base image as the images to be registered, and the number of images to be registered is 80% of the number of total images.
    vector<double> similartobaseImg_temp;
    for (int k = 0; k < vecInImgs.size(); k++){
        double temp;
        Mat imgtemp = vecInImgs[k];
        temp = mat_corr(outImg,imgtemp);
        similartobaseImg_temp.push_back(temp);
    }

    vector<size_t> similartobaseImg_temp_idx = sort_indexes_e(similartobaseImg_temp);
    cout << "The images for registration: ";
    for (int i = vecInImgs.size() - 0.8 * vecInImgs.size(); i < vecInImgs.size()-1; i++){
        Mat mat_temp;
        vecInImgs[similartobaseImg_temp_idx[i]].convertTo(mat_temp, CV_8UC1);
        if (!mat_temp.empty())
            vecoutImgs.push_back(mat_temp);
        cout << similartobaseImg_temp_idx[i] << ' ';
    }
    cout << endl;
}

// Registration.
void ImageRegistration(const std::vector<cv::Mat>& vecInImgs, cv::Mat refImg, std::vector<cv::Mat>& vecoutImgs, std::vector<double> coefficientAfterAlignment){
    // image parameter
    int row = refImg.rows;
    int column = refImg.cols;
    int bscan_number = vecInImgs.size();

    // pixel subdivision parameter
    int q = 1;

    //parameters for vertical alignment
    int N = 25*q; // vertical ranges of registration in every column
    int p = 1*q; // continiousity range of every registration
    int M = 2; // margin when pixels are outside the designated region
    double lamda = 0.1; // Regularisation parameter

    // parameters for horizontal alignment
    int R = 32; // the range of the horizontal alignment is 32.

    for (int k = 0; k < bscan_number; k++){
        // calculate the correlation coefficient of the inImg and the refImg
        Mat inImg= vecInImgs[k];
        double r_raw = mat_corr(inImg, refImg);
        cout << "r_raw(" << k << ") = " << r_raw << endl;

        /// alignment in the initial place
        // calculate the negative correlation coeffieient kept in E[][]
        vector<vector<double> >E((2*N+1+2*M),vector<double>(column)); // 动态定义二维数组 (2*N+1+2*M)行 column列
        for (int n = 0; n < column; n++){
            for (int u = 0; u < M; u++) {
                E[u][n] = 100;
                E[2*N+1+M+u][n] = 100;
            } // set the margin to the infinity
            Mat img_col = inImg.col(n).clone();
            Mat ref_col = refImg.col(n).clone();
            Mat img_drift;
            for (int i = -N; i <= N; i++){
                circShift(img_col,img_drift, i, 0);
                E[i+M+N][n] = - mat_corr(ref_col, img_drift);
            }
        }

        // calculate the dynamic programming equation kept in d[][][]
        vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
        // the initialization equation: n = 1;
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                d[i+N][k+p][1] = E[i+N+M][1] + E[i+N+M+k][0];
            }
        }

        // the recursion formula for the dynamic programming routine
        vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
        vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
        vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
        for (int n = 2; n < column; n++){
            for (int i = -N; i <= N; i++){
                for (int k = -p; k <= p; k++){
                    if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                        for (int l = -p; l <= p; l++){
                            cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                        }
                        vector<double> vec_temp(2*p+1);
                        for (int u = 0; u < (2*p+1); u++){
                            vec_temp[u] = cost[i+k+N][u][n];
                        }
                        cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                        l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                        d[i+N][k+p][n] = E[i+N+M][n] + cost_min[i+k+N][k+p][n];
                    }
                }
            }
            for (int x = 0; x < (2*N+1); x++){
                for (int y = 0; y < (2*p+1); y++){
                    for (int z = 0; z < column; z++){
                        if (d[x][y][z] == 0)  d[x][y][z] = 100;
                    }
                }
            }
        }

        // backtracking
        vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
        vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
        for (int k = 0; k < (2*p+1); k++){
            vector<double> vec_temp(2*N+1);
            for (int u = 0; u < (2*N+1); u++){
                vec_temp[u] = d[u][k][column-1];
            }
            dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
            ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
        }
        double d_min = *min_element(dn_min.begin(),dn_min.end());
        double k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
        vector <double> i_min(column);
        i_min[column-1] = ii_min[k_min-1];
        vector <int> drift(column); // keep the drift of every column in the image
        drift[column-1] = i_min[column-1] - N - 1;
        vector <int> km(column);
        km[column-1] = k_min - p - 1;
        i_min[column-2] = i_min[column-1] + km[column-1];
        drift[column-2] = i_min[column-2] - N - 1;
        for (int n = column -2; n > 0; n--){
            if(i_min[n+1] < 2) i_min[n+1] = 2;
            if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
            km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
            i_min[n-1] = i_min[n] + km[n];
            drift[n-1] = i_min[n-1] - N - 1;
        }

        // do the correction
        Mat align0 = Mat(inImg.size(), CV_8UC1);
        vector<vector<double> >E2((2*N+1+2*M),vector<double>(column));
        for (int n = 0; n < column; n++){
            Mat img_col = inImg.col(n).clone();
            Mat img_drift;
            circShift(img_col,img_drift, drift[n], 0);
            img_drift.copyTo(align0.col(n));
        }
        double value0 = mat_corr(align0,refImg);
        cout << "value0 = " << value0 << endl;
//        string Img_Name = "DPalign"+to_string(k)+".jpg";
//        imwrite(Img_Name,align0);

        /// alignment in the right place
        Mat align1 = Mat(inImg.size(), CV_8UC1);
        circShift(inImg, align1,0, R); // move R pixels on the right
        double value1 = mat_corr(align1,refImg);
        cout <<  "value1 = " << value1 << endl;

        /// alignment in the left place
        Mat align2 = Mat(inImg.size(), CV_8UC1);
        circShift(inImg, align2,0, -R); // move R pixels on the right
        double value2 = mat_corr(align2,refImg);
        cout <<  "value2 = " << value2 << endl;

        /// binary search to find the optimal horizontal drift
        vector <double> value_compare(3);
        value_compare[0] = value0; value_compare[1] = value1; value_compare[2] = value2;
        int p1_max,p1_sec,p2_max,p2_sec,p_sec;
        double v1_max,v1_sec,v2_max,v2_sec,value_sec;
        v1_max = *max_element(value_compare.begin(),value_compare.end());
        cout << "v1_max = " << v1_max << endl;
        if(v1_max == value0){
            if(value1 > value2){
                p1_max = 0;
                p1_sec = R;
                v1_sec = value1;
            }
            else{
                p1_max = 0;
                p1_sec = -R;
                v1_sec = value2;
            }
        }

        if(v1_max == value0) {
            p2_max = p1_max;
            p2_sec = p1_sec;
            v2_max = v1_max;
            v2_sec = v1_sec;
            Mat align_sec = Mat(inImg.size(), CV_8UC1);
            Mat align_max = Mat(inImg.size(), CV_8UC1);
            align0.copyTo(align_max);
            while (abs(p2_max - p2_sec) > 1) {
                p_sec = (p2_max + p2_sec) / 2;
                circShift(inImg, align_sec, 0, p_sec); // move R pixels on the right
                for (int n = 0; n < column; n++) {
                    for (int u = 0; u < M; u++) {
                        E[u][n] = 100;
                        E[2 * N + 1 + M + u][n] = 100;
                    } // set the margin to the infinity
                    Mat img_col = align_sec.col(n).clone();
                    Mat ref_col = refImg.col(n).clone();
                    Mat img_drift;
                    for (int i = -N; i <= N; i++) {
                        circShift(img_col, img_drift, i, 0);
                        E[i + M + N][n] = -mat_corr(ref_col, img_drift);
                    }
                }

                // the initialization equation: n = 1;
                for (int i = -N; i <= N; i++) {
                    for (int k = -p; k <= p; k++) {
                        d[i + N][k + p][1] = E[i + N + M][1] + E[i + N + M + k][0];
                    }
                }

                // the recursion formula for the dynamic programming routine
                for (int n = 2; n < column; n++) {
                    for (int i = -N; i <= N; i++) {
                        for (int k = -p; k <= p; k++) {
                            if ((i + k + N) >= 0 && ((i + k + N) <= (2 * N))) { // limit the boundry of d[][][]
                                for (int l = -p; l <= p; l++) {
                                    cost[i + k + N][l + p][n] = d[i + k + N][l + p][n - 1] + lamda * (l - k) * (l - k);
                                }
                                vector<double> vec_temp(2 * p + 1);
                                for (int u = 0; u < (2 * p + 1); u++) {
                                    vec_temp[u] = cost[i + k + N][u][n];
                                }
                                cost_min[i + k + N][k + p][n] = *min_element(vec_temp.begin(), vec_temp.end());
                                l_min[i + k + N][k + p][n] =
                                        min_element(vec_temp.begin(), vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                                d[i + N][k + p][n] = E[i + N + M][n] + cost_min[i + k + N][k + p][n];
                            }
                        }
                    }
                    for (int x = 0; x < (2 * N + 1); x++) {
                        for (int y = 0; y < (2 * p + 1); y++) {
                            for (int z = 0; z < column; z++) {
                                if (d[x][y][z] == 0) d[x][y][z] = 100;
                            }
                        }
                    }
                }

                // backtracking
                for (int k = 0; k < (2 * p + 1); k++) {
                    vector<double> vec_temp(2 * N + 1);
                    for (int u = 0; u < (2 * N + 1); u++) {
                        vec_temp[u] = d[u][k][column - 1];
                    }
                    dn_min[k] = *min_element(vec_temp.begin(), vec_temp.end());
                    ii_min[k] = min_element(vec_temp.begin(), vec_temp.end()) - vec_temp.begin() + 1;
                }
                double d_min = *min_element(dn_min.begin(), dn_min.end());
                double k_min = min_element(dn_min.begin(), dn_min.end()) - dn_min.begin() + 1;
                i_min[column - 1] = ii_min[k_min - 1];
                drift[column - 1] = i_min[column - 1] - N - 1;
                km[column - 1] = k_min - p - 1;
                i_min[column - 2] = i_min[column - 1] + km[column - 1];
                drift[column - 2] = i_min[column - 2] - N - 1;
                for (int n = column - 2; n > 0; n--) {
                    if (i_min[n + 1] < 2) i_min[n + 1] = 2;
                    if (i_min[n + 1] > 2*N) i_min[n + 1] = 2*N; // keep the boundry
                    km[n] = l_min[i_min[n + 1] - 1][km[n + 1] + p][n + 1] - p - 1;
                    i_min[n - 1] = i_min[n] + km[n];
                    drift[n - 1] = i_min[n - 1] - N - 1;
                }

                // do the correction
                for (int n = 0; n < column; n++) {
                    Mat img_col = align_sec.col(n).clone();
                    Mat img_drift;
                    circShift(img_col, img_drift, drift[n], 0);
                    img_drift.copyTo(align_sec.col(n));
                }
                double value_sec = mat_corr(align_sec, refImg);

                /// interation
                if (v2_max > value_sec) {
                    v2_sec = value_sec;
                    p2_sec = p_sec;
                }
                else {
                    v2_sec = v2_max;
                    p2_sec = p2_max;
                    v2_max = value_sec;
                    p2_max = p_sec;
                    align_sec.copyTo(align_max);
                }
                cout << "p2_max = " << p2_max << "; v2_max = " << v2_max << "; p2_sec = " << p2_sec << "; v2_sec = "
                     << v2_sec << endl;
            }
            double similarity; // similarity after alignment
            double similarity2;
            Mat outImg = Mat(inImg.size(), CV_8UC1);
            align_max.copyTo(outImg);
            similarity = v2_max;
            cout << "similarity = " << similarity << endl;
            string Img_Name = "registration"+to_string(k)+".jpg";
            imwrite(Img_Name,outImg);
            if (!outImg.empty()){
                vecoutImgs.push_back(outImg);
                coefficientAfterAlignment.push_back(similarity);
            }
        }
    }
}

void ImageRegistration_s(const std::vector<cv::Mat>& vecInImgs, cv::Mat refImg, std::vector<cv::Mat>& vecoutImgs, std::vector<double> coefficientAfterAlignment){
    // image parameter
    int row = refImg.rows;
    int column = refImg.cols/2;
    int bscan_number = vecInImgs.size();

    // pixel subdivision parameter
    int q = 1;

    //parameters for vertical alignment
    int N = 25*q; // vertical ranges of registration in every column
    int p = 1*q; // continiousity range of every registration
    int M = 2; // margin when pixels are outside the designated region
    double lamda = 0.1; // Regularisation parameter

    // parameters for horizontal alignment
    int R = 32; // the range of the horizontal alignment is 32.
    Mat refImg_s = Mat::zeros(row, column, CV_8UC1);
    for (int n = 0; n < column; n++){
        refImg.col(2*n).copyTo(refImg_s.col(n));
    }

    for (int k = 0; k < bscan_number; k++){
        // calculate the correlation coefficient of the inImg and the refImg
        Mat tImg = vecInImgs[k];
        Mat inImg = Mat::zeros(row, column, CV_8UC1);
        for (int n = 0; n < column; n++){
            tImg.col(2*n).copyTo(inImg.col(n));
        }
        double r_raw = mat_corr(tImg, refImg);
        cout << "r_raw(" << k << ") = " << r_raw << endl;

        /// alignment in the initial place
        // calculate the negative correlation coeffieient kept in E[][]
        vector<vector<double> >E((2*N+1+2*M),vector<double>(column)); // 动态定义二维数组 (2*N+1+2*M)行 column列
        for (int n = 0; n < column; n++){
            for (int u = 0; u < M; u++) {
                E[u][n] = 100;
                E[2*N+1+M+u][n] = 100;
            } // set the margin to the infinity
            Mat img_col = inImg.col(n).clone();
            Mat ref_col = refImg_s.col(n).clone();
            Mat img_drift;
            for (int i = -N; i <= N; i++){
                circShift(img_col,img_drift, i, 0);
                E[i+M+N][n] = - mat_corr(ref_col, img_drift);
            }
        }

        // calculate the dynamic programming equation kept in d[][][]
        vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
        // the initialization equation: n = 1;
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                d[i+N][k+p][1] = E[i+N+M][1] + E[i+N+M+k][0];
            }
        }

        // the recursion formula for the dynamic programming routine
        vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
        vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
        vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
        for (int n = 2; n < column; n++){
            for (int i = -N; i <= N; i++){
                for (int k = -p; k <= p; k++){
                    if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                        for (int l = -p; l <= p; l++){
                            cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                        }
                        vector<double> vec_temp(2*p+1);
                        for (int u = 0; u < (2*p+1); u++){
                            vec_temp[u] = cost[i+k+N][u][n];
                        }
                        cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                        l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                        d[i+N][k+p][n] = E[i+N+M][n] + cost_min[i+k+N][k+p][n];
                    }
                }
            }
            for (int x = 0; x < (2*N+1); x++){
                for (int y = 0; y < (2*p+1); y++){
                    for (int z = 0; z < column; z++){
                        if (d[x][y][z] == 0)  d[x][y][z] = 100;
                    }
                }
            }
        }

        // backtracking
        vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
        vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
        for (int k = 0; k < (2*p+1); k++){
            vector<double> vec_temp(2*N+1);
            for (int u = 0; u < (2*N+1); u++){
                vec_temp[u] = d[u][k][column-1];
            }
            dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
            ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
        }
        double d_min = *min_element(dn_min.begin(),dn_min.end());
        double k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
        vector <double> i_min(column);
        i_min[column-1] = ii_min[k_min-1];
        vector <int> drift(column); // keep the drift of every column in the image
        drift[column-1] = i_min[column-1] - N - 1;
        vector <int> km(column);
        km[column-1] = k_min - p - 1;
        i_min[column-2] = i_min[column-1] + km[column-1];
        drift[column-2] = i_min[column-2] - N - 1;
        for (int n = column -2; n > 0; n--){
            if(i_min[n+1] < 2) i_min[n+1] = 2;
            if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
            km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
            i_min[n-1] = i_min[n] + km[n];
            drift[n-1] = i_min[n-1] - N - 1;
        }

        // do the correction
        Mat align0 = Mat(tImg.size(), CV_8UC1);
        vector<vector<double> >E2((2*N+1+2*M),vector<double>(column));
        for (int n = 0; n < column; n++){
            Mat img_col = tImg.colRange(2*n, 2*n+2).clone();
            Mat img_drift;
            circShift(img_col,img_drift, drift[n], 0);
            img_drift.copyTo(align0.colRange(2*n, 2*n+2));
        }
        double value0 = mat_corr(align0,refImg);
        cout << "value0 = " << value0 << endl;
//        string Img_Name = "DPalign"+to_string(k)+".jpg";
//        imwrite(Img_Name,align0);

        /// alignment in the right place
        Mat align1 = Mat(tImg.size(), CV_8UC1);
        circShift(tImg, align1,0, R); // move R pixels on the right
        double value1 = mat_corr(align1,refImg);
        cout <<  "value1 = " << value1 << endl;

        /// alignment in the left place
        Mat align2 = Mat(tImg.size(), CV_8UC1);
        circShift(tImg, align2,0, -R); // move R pixels on the right
        double value2 = mat_corr(align2,refImg);
        cout <<  "value2 = " << value2 << endl;

        /// binary search to find the optimal horizontal drift
        vector <double> value_compare(3);
        value_compare[0] = value0; value_compare[1] = value1; value_compare[2] = value2;
        int p1_max,p1_sec,p2_max,p2_sec,p_sec;
        double v1_max,v1_sec,v2_max,v2_sec,value_sec;
        v1_max = *max_element(value_compare.begin(),value_compare.end());
        cout << "v1_max = " << v1_max << endl;
        if(v1_max == value0){
            if(value1 > value2){
                p1_max = 0;
                p1_sec = R;
                v1_sec = value1;
            }
            else{
                p1_max = 0;
                p1_sec = -R;
                v1_sec = value2;
            }
        }

        if(v1_max == value0) {
            p2_max = p1_max;
            p2_sec = p1_sec;
            v2_max = v1_max;
            v2_sec = v1_sec;
            Mat align_sec = Mat(tImg.size(), CV_8UC1);
            Mat align_max = Mat(tImg.size(), CV_8UC1);
            align0.copyTo(align_max);
            while (abs(p2_max - p2_sec) > 1) {
                p_sec = (p2_max + p2_sec) / 2;
                circShift(tImg, align_sec, 0, p_sec); // move R pixels on the right
                Mat align_sec_s = Mat::zeros(row, column, CV_8UC1);
                for (int n = 0; n < column; n++){
                    align_sec.col(2*n).copyTo(align_sec_s.col(n));
                }
                for (int n = 0; n < column; n++) {
                    for (int u = 0; u < M; u++) {
                        E[u][n] = 100;
                        E[2 * N + 1 + M + u][n] = 100;
                    } // set the margin to the infinity
                    Mat img_col = align_sec_s.col(n).clone();
                    Mat ref_col = refImg_s.col(n).clone();
                    Mat img_drift;
                    for (int i = -N; i <= N; i++) {
                        circShift(img_col, img_drift, i, 0);
                        E[i + M + N][n] = -mat_corr(ref_col, img_drift);
                    }
                }

                // the initialization equation: n = 1;
                for (int i = -N; i <= N; i++) {
                    for (int k = -p; k <= p; k++) {
                        d[i + N][k + p][1] = E[i + N + M][1] + E[i + N + M + k][0];
                    }
                }

                // the recursion formula for the dynamic programming routine
                for (int n = 2; n < column; n++) {
                    for (int i = -N; i <= N; i++) {
                        for (int k = -p; k <= p; k++) {
                            if ((i + k + N) >= 0 && ((i + k + N) <= (2 * N))) { // limit the boundry of d[][][]
                                for (int l = -p; l <= p; l++) {
                                    cost[i + k + N][l + p][n] = d[i + k + N][l + p][n - 1] + lamda * (l - k) * (l - k);
                                }
                                vector<double> vec_temp(2 * p + 1);
                                for (int u = 0; u < (2 * p + 1); u++) {
                                    vec_temp[u] = cost[i + k + N][u][n];
                                }
                                cost_min[i + k + N][k + p][n] = *min_element(vec_temp.begin(), vec_temp.end());
                                l_min[i + k + N][k + p][n] =
                                        min_element(vec_temp.begin(), vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                                d[i + N][k + p][n] = E[i + N + M][n] + cost_min[i + k + N][k + p][n];
                            }
                        }
                    }
                    for (int x = 0; x < (2 * N + 1); x++) {
                        for (int y = 0; y < (2 * p + 1); y++) {
                            for (int z = 0; z < column; z++) {
                                if (d[x][y][z] == 0) d[x][y][z] = 100;
                            }
                        }
                    }
                }

                // backtracking
                for (int k = 0; k < (2 * p + 1); k++) {
                    vector<double> vec_temp(2 * N + 1);
                    for (int u = 0; u < (2 * N + 1); u++) {
                        vec_temp[u] = d[u][k][column - 1];
                    }
                    dn_min[k] = *min_element(vec_temp.begin(), vec_temp.end());
                    ii_min[k] = min_element(vec_temp.begin(), vec_temp.end()) - vec_temp.begin() + 1;
                }
                double d_min = *min_element(dn_min.begin(), dn_min.end());
                double k_min = min_element(dn_min.begin(), dn_min.end()) - dn_min.begin() + 1;
                i_min[column - 1] = ii_min[k_min - 1];
                drift[column - 1] = i_min[column - 1] - N - 1;
                km[column - 1] = k_min - p - 1;
                i_min[column - 2] = i_min[column - 1] + km[column - 1];
                drift[column - 2] = i_min[column - 2] - N - 1;
                for (int n = column - 2; n > 0; n--) {
                    if (i_min[n + 1] < 2) i_min[n + 1] = 2;
                    if (i_min[n + 1] > 2*N) i_min[n + 1] = 2*N; // keep the boundry
                    km[n] = l_min[i_min[n + 1] - 1][km[n + 1] + p][n + 1] - p - 1;
                    i_min[n - 1] = i_min[n] + km[n];
                    drift[n - 1] = i_min[n - 1] - N - 1;
                }

                // do the correction
                for (int n = 0; n < column; n++) {
                    Mat img_col = align_sec.colRange(2*n, 2*n+2).clone();
                    Mat img_drift;
                    circShift(img_col, img_drift, drift[n], 0);
                    img_drift.copyTo(align_sec.colRange(2*n, 2*n+2));
                }
                double value_sec = mat_corr(align_sec, refImg);

                /// interation
                if (v2_max > value_sec) {
                    v2_sec = value_sec;
                    p2_sec = p_sec;
                }
                else {
                    v2_sec = v2_max;
                    p2_sec = p2_max;
                    v2_max = value_sec;
                    p2_max = p_sec;
                    align_sec.copyTo(align_max);
                }
                cout << "p2_max = " << p2_max << "; v2_max = " << v2_max << "; p2_sec = " << p2_sec << "; v2_sec = "
                     << v2_sec << endl;
            }
            double similarity; // similarity after alignment
            double similarity2;
            Mat outImg = Mat(tImg.size(), CV_8UC1);
            align_max.copyTo(outImg);
            similarity = v2_max;
            cout << "similarity = " << similarity << endl;
            string Img_Name = "registration"+to_string(k)+".jpg";
            imwrite(Img_Name,outImg);
            if (!outImg.empty()){
                vecoutImgs.push_back(outImg);
                coefficientAfterAlignment.push_back(similarity);
            }
        }
    }
}

void image2register(cv::Mat& inImg, cv::Mat& refImg, cv::Mat& outImg){
    // image parameter
    int row = refImg.rows;
    int column = refImg.cols;

    // pixel subdivision parameter
    int q = 1;

    //parameters for vertical alignment
    int N = 25*q; // vertical ranges of registration in every column
    int p = 1*q; // continiousity range of every registration
    int M = 2; // margin when pixels are outside the designated region, notice that p must <= M
    double lamda = 0.1; // Regularisation parameter

    // parameters for horizontal alignment
    int R = 16; // the range of the horizontal alignment is 32.

    // calculate the correlation coefficient of the inImg and the refImg
    double r_raw = mat_corr(inImg, refImg);
    cout << "r_raw = " << r_raw  << endl;

    double similarity; // similarity after alignment
    /// alignment in the initial place
    // calculate the negative correlation coeffieient kept in E[][]
    vector<vector<double> >E((2*N+1+2*M),vector<double>(column)); // 动态定义二维数组 (2*N+1+2*M)行 column列
    for (int n = 0; n < column; n++){
        for (int u = 0; u < M; u++) {
            E[u][n] = 100;
            E[2*N+1+M+u][n] = 100;
        } // set the margin to the infinity
        Mat img_col = inImg.col(n).clone();
        Mat ref_col = refImg.col(n).clone();
        Mat img_drift;
        for (int i = -N; i <= N; i++){
            circShift(img_col,img_drift, i, 0);
            E[i+M+N][n] = - mat_corr(ref_col, img_drift);
        }
    }

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = E[i+N+M][1] + E[i+N+M+k][0];
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = E[i+N+M][n] + cost_min[i+k+N][k+p][n];
                }
            }
        }
        for (int x = 0; x < (2*N+1); x++){
            for (int y = 0; y < (2*p+1); y++){
                for (int z = 0; z < column; z++){
                    if (d[x][y][z] == 0)  d[x][y][z] = 100;
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
    double d_min = *min_element(dn_min.begin(),dn_min.end());
    double k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    vector <int> drift(column); // keep the drift of every column in the image
    drift[column-1] = i_min[column-1] - N - 1;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    drift[column-2] = i_min[column-2] - N - 1;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        drift[n-1] = i_min[n-1] - N - 1;
    }

    // do the correction
    Mat align0 = Mat(inImg.size(), CV_8UC1);
    vector<vector<double> >E2((2*N+1+2*M),vector<double>(column));
    for (int n = 0; n < column; n++){
        Mat img_col = inImg.col(n).clone();
        Mat img_drift;
        circShift(img_col,img_drift, drift[n], 0);
        img_drift.copyTo(align0.col(n));
    }
    double value0 = mat_corr(align0,refImg);
    cout << "value0 = " << value0 << endl;

    /// alignment in the right place
    Mat align1 = Mat(inImg.size(), CV_8UC1);
    circShift(inImg, align1,0, R); // move R pixels on the right
    double value1 = mat_corr(align1,refImg);
    cout <<  "value1 = " << value1 << endl;

    /// alignment in the left place
    Mat align2 = Mat(inImg.size(), CV_8UC1);
    circShift(inImg, align2,0, -R); // move R pixels on the right
    double value2 = mat_corr(align2,refImg);
    cout <<  "value2 = " << value2 << endl;

    /// binary search to find the optimal horizontal drift
    vector <double> value_compare(3);
    value_compare[0] = value0; value_compare[1] = value1; value_compare[2] = value2;
    int p1_max,p1_sec,p2_max,p2_sec,p_sec;
    double v1_max,v1_sec,v2_max,v2_sec,value_sec;
    v1_max = *max_element(value_compare.begin(),value_compare.end());
    cout << "v1_max = " << v1_max << endl;
    if(v1_max == value0){
        if(value1 > value2){
            p1_max = 0;
            p1_sec = R;
            v1_sec = value1;
        }
        else{
            p1_max = 0;
            p1_sec = -R;
            v1_sec = value2;
        }
    }

    if(v1_max == value0){
        p2_max = p1_max;
        p2_sec = p1_sec;
        v2_max = v1_max;
        v2_sec = v1_sec;
        Mat align_sec = Mat(inImg.size(), CV_8UC1);
        Mat align_max = Mat(inImg.size(), CV_8UC1);
        align0.copyTo(align_max);
        while(abs(p2_max - p2_sec) > 1){
            p_sec = (p2_max + p2_sec)/2;
            circShift(inImg, align_sec,0, p_sec); // move R pixels on the right
            for (int n = 0; n < column; n++){
                for (int u = 0; u < M; u++) {
                    E[u][n] = 100;
                    E[2*N+1+M+u][n] = 100;
                } // set the margin to the infinity
                Mat img_col = align_sec.col(n).clone();
                Mat ref_col = refImg.col(n).clone();
                Mat img_drift;
                for (int i = -N; i <= N; i++){
                    circShift(img_col,img_drift, i, 0);
                    E[i+M+N][n] = - mat_corr(ref_col, img_drift);
                }
            }

            // the initialization equation: n = 1;
            for (int i = -N; i <= N; i++){
                for (int k = -p; k <= p; k++){
                    d[i+N][k+p][1] = E[i+N+M][1] + E[i+N+M+k][0];
                }
            }

            // the recursion formula for the dynamic programming routine
            for (int n = 2; n < column; n++){
                for (int i = -N; i <= N; i++){
                    for (int k = -p; k <= p; k++){
                        if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                            for (int l = -p; l <= p; l++){
                                cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                            }
                            vector<double> vec_temp(2*p+1);
                            for (int u = 0; u < (2*p+1); u++){
                                vec_temp[u] = cost[i+k+N][u][n];
                            }
                            cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                            l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                            d[i+N][k+p][n] = E[i+N+M][n] + cost_min[i+k+N][k+p][n];
                        }
                    }
                }
                for (int x = 0; x < (2*N+1); x++){
                    for (int y = 0; y < (2*p+1); y++){
                        for (int z = 0; z < column; z++){
                            if (d[x][y][z] == 0)  d[x][y][z] = 100;
                        }
                    }
                }
            }

            // backtracking
            for (int k = 0; k < (2*p+1); k++){
                vector<double> vec_temp(2*N+1);
                for (int u = 0; u < (2*N+1); u++){
                    vec_temp[u] = d[u][k][column-1];
                }
                dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
                ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
            }
            double d_min = *min_element(dn_min.begin(),dn_min.end());
            double k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
            i_min[column-1] = ii_min[k_min-1];
            drift[column-1] = i_min[column-1] - N - 1;
            km[column-1] = k_min - p - 1;
            i_min[column-2] = i_min[column-1] + km[column-1];
            drift[column-2] = i_min[column-2] - N - 1;
            for (int n = column -2; n > 0; n--){
                if(i_min[n+1] < 2) i_min[n+1] = 2;
                if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
                km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
                i_min[n-1] = i_min[n] + km[n];
                drift[n-1] = i_min[n-1] - N - 1;
            }

            // do the correction
            for (int n = 0; n < column; n++){
                Mat img_col = align_sec.col(n).clone();
                Mat img_drift;
                circShift(img_col,img_drift, drift[n], 0);
                img_drift.copyTo(align_sec.col(n));
            }
            double value_sec = mat_corr(align_sec,refImg);

            /// interation
            if (v2_max > value_sec) {
                v2_sec = value_sec;
                p2_sec = p_sec;
            }
            else {
                v2_sec = v2_max;
                p2_sec = p2_max;
                v2_max = value_sec;
                p2_max = p_sec;
                align_sec.copyTo(align_max);
            }
            cout << "p2_max = " << p2_max << "; v2_max = " << v2_max << "; p2_sec = " << p2_sec << "; v2_sec = " << v2_sec << endl;
        }

        align_max.copyTo(outImg);
        similarity = v2_max;
        cout << "similarity = " << similarity << endl;
        imshow("out",outImg);
        waitKey();
    }
}

// Remove the ghost image. True: Imgae has no ghost. False:Image has ghost when averaging.
bool RemoveGhostImage(cv::Mat& inImg, cv::Mat& fltRef){
    // parameters
    double p1 = 0.91;
    double p2 = 0.97;
    int t = 20;

    Mat fltImg;
    medianBlur(inImg, fltImg, 7);

    Mat diff;
    absdiff(fltImg, fltRef, diff);
    Mat hist;
    int histSize[] = { 256 };
    int channels = 0;
    float midRanges[] = { 0, 256 };
    const float *ranges[] = { midRanges };
    calcHist(&diff,1,&channels ,Mat(),hist,1,histSize,ranges,true,false);

    Mat pdf = Mat(hist.size(),CV_32FC1);
    pdf = hist/(inImg.cols * inImg.rows);
    vector <double> pdfvec = convertMat2Vector<double>(pdf);
    vector <double> cdfvec(256);
    for (int i = 0; i <= 255; i++) {
        for (int j = 0; j<= i; j++)
            cdfvec[i] += pdfvec[j];
    }

    vector <double> closetop1(256);
    for (int i = 0; i <= 255; i++) {
        closetop1[i] = abs(cdfvec[i] - p1);
    }
    int thresh1 = min_element(closetop1.begin(), closetop1.end()) - closetop1.begin();
    cout << "thresh1 = " << thresh1 << endl;

    vector <double> closetop2(256);
    for (int i = 0; i <= 255; i++) {
        closetop2[i] = abs(cdfvec[i] - p2);
    }
    int thresh2 = min_element(closetop2.begin(), closetop2.end()) - closetop2.begin();
    cout << "thresh2 = " << thresh2 << endl;

    int thresh = thresh2 - thresh1;
    cout << "thresh2 - thresh1 = " << thresh << endl;

//    return true;
    if (thresh > t) return false;
    else return true;
}

void findRPE(const cv::Mat& inImg, vector<int>& RPE){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    // parameters
    int N = row/2 - 50; // vertical ranges
    int p = 3;   // threshold of jump pixels
    double lamda = 10; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (fltImg.at<uchar>(i+N+row/2-N, 1) + fltImg.at<uchar>(i+N+k+row/2-N, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - fltImg.at<uchar>(i+N+row/2-N, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    RPE[column-1] = i_min[column-1] - 1 + row/2 - N;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    RPE[column-2] = i_min[column-2] - N - 1 + row/2;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        RPE[n-1] = i_min[n-1] - N - 1 + row/2;
    }
}

void findONL(const cv::Mat& inImg, vector<int>& RPE, vector<int>& ONL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 7);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = RPE[j]-8; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = 0; i < RPE[j] - 30; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS", YGradientABS);
//    waitKey();

    int RPELoc = *max_element(RPE.begin(),RPE.end());
    // parameters
    int N = (RPELoc - 50)/2; // vertical ranges: 2*N+1
    int p = 3;   // threshold of jump pixels
    double lamda = 10; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    ONL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    ONL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        ONL[n-1] = i_min[n-1] - 1 + 50;
    }
}

void Segmentation(const cv::Mat& inImg, vector<double>& ONLvec) {
    int column = inImg.cols;
    vector <int> layerRPE(column);
    findRPE(inImg, layerRPE);
    vector<double> layerRPE_double(layerRPE.begin(), layerRPE.end());
    layerRPE_double = sg_smooth(layerRPE_double, 5, 0);

    vector <int> layerONL(column);
    findONL(inImg, layerRPE, layerONL);
    for (int n = 0; n < column; ++n) {
        if ((layerRPE[n] - layerONL[n]) < 0) layerONL[n] = layerRPE[n];
    }
    vector<double> layerONL_double(layerONL.begin(), layerONL.end());
    ONLvec = sg_smooth(layerONL_double, 5, 0);
    Mat image;
    cvtColor(inImg, image, COLOR_GRAY2BGR);
    for (int n = 0; n < image.cols; n++) {
        Point onl(n, ONLvec[n]);
        circle(image, onl, 1, Scalar(255, 245, 0), -1); //淡蓝
        Point rpe(n, layerRPE_double[n]);
        circle(image, rpe, 1, Scalar(0, 255, 255), -1); //黄色
    }
    //imshow("segmentation", image);
    //waitKey();
    imwrite("segmentated.jpg", image);
}
// Enhance Bscan image area.
void EnhanceBscanArea(const cv::Mat& inImg, cv::Mat& outImg){
    vector<double> ONLLayer;
    Segmentation(inImg, ONLLayer);
    vector<int> layerONL(ONLLayer.begin(), ONLLayer.end());
    //    medianBlur(inImg, outImg, 3);
    //    GaussianBlur(inImg, tempImg, Size(3, 3), 3, 3);
    // Mat tempImg;
    // WienerFilter(inImg, tempImg, Size(5, 5));
    inImg.convertTo(outImg, CV_32FC1);

    Mat hist;
    int histSize[] = { 256 };
    int channels = 0;
    float midRanges[] = { 0, 256 };
    const float *ranges[] = { midRanges };
    calcHist(&outImg, 1, &channels, Mat(), hist, 1, histSize, ranges, true, false);

    Mat pdf = Mat(hist.size(), CV_32FC1);
    pdf = hist / (inImg.cols * inImg.rows);
    vector <double> pdfvec = convertMat2Vector<double>(pdf);
    double pdf_max, pdf_min;
    double alpha = 0.01; /// the adjusted parameter
    minMaxIdx(pdf, &pdf_min, &pdf_max, 0, 0);
    double pdf_diff = pdf_max - pdf_min;
    double * pdf_w = new double[256];
#pragma omp parallel for
    for (int i = 0; i <= 255; i++) {
        pdf_w[i] = pdf_max * pow(((pdfvec[i] - pdf_min) / pdf_diff), alpha);
    }
    double sum_pdf_w = 0;
    for (int i = 0; i <= 255; i++) {
        sum_pdf_w += pdf_w[i];
    }

    double * cdf_w = new double[256];
    //    vector <double> cdf_w(256);
#pragma omp parallel for
    for (int i = 0; i <= 255; i++) {
        pdf_w[i] = pdf_w[i] / sum_pdf_w;
        cdf_w[i] = pdf_w[i];
    }
    for (int i = 1; i <= 255; i++) {
        cdf_w[i] = cdf_w[i - 1] + pdf_w[i];
    }

    double value_max, OCT_Min;
    minMaxLoc(outImg, NULL, &value_max, NULL, NULL);
#pragma omp parallel for
    for (int l = 0; l < 255; l++) {
        float graydivide = float(l) / 255;
        for (int j = 0; j < outImg.cols; ++j) {
            for (int i = 0; i < layerONL[j]; ++i) {
                if (outImg.at<float>(i, j) == float(l)) {
                    outImg.at<float>(i, j) = float(value_max * pow(graydivide, (1 - cdf_w[l])));
                }
            }
        }
    }

    outImg.convertTo(outImg, CV_8UC1);
}

// Enhance Bscan image.
void EnhanceBscan(const cv::Mat& inImg, cv::Mat& outImg){
    Mat tempImg;
//    medianBlur(inImg, outImg, 3);
//    GaussianBlur(inImg, tempImg, Size(3, 3), 3, 3);
    WienerFilter(inImg, tempImg, Size(5, 5));
    tempImg.convertTo(outImg, CV_32FC1);
//    inImg.convertTo(outImg, CV_32FC1);
    Mat hist;
    int histSize[] = { 256 };
    int channels = 0;
    float midRanges[] = { 0, 256 };
    const float *ranges[] = { midRanges };
    calcHist(&outImg,1,&channels ,Mat(),hist,1,histSize,ranges,true,false);

    Mat pdf = Mat(hist.size(),CV_32FC1);
    pdf = hist/(inImg.cols * inImg.rows);
    vector <double> pdfvec = convertMat2Vector<double>(pdf);
    double pdf_max,pdf_min;
    double alpha = 0.01; /// the adjusted parameter
    minMaxIdx(pdf, &pdf_min, &pdf_max, 0, 0);
    double pdf_diff = pdf_max - pdf_min;
    double * pdf_w = new double[256];
    for (int i = 0; i <= 255; i++) {
        pdf_w[i] = pdf_max * pow(((pdfvec[i] - pdf_min)/pdf_diff), alpha);
    }
    double sum_pdf_w;
    for (int i = 0; i <= 255; i++) {
        sum_pdf_w += pdf_w[i];
    }

    double * cdf_w = new double[256];
    //    vector <double> cdf_w(256);
    for (int i = 0; i <= 255; i++) {
        pdf_w[i] = pdf_w[i]/sum_pdf_w;
        cdf_w[i] = pdf_w[i];
    }
    for (int i = 1; i <= 255; i++) {
        cdf_w[i] = cdf_w[i-1] + pdf_w[i];
    }

//    double value_max,value_min;
//    minMaxIdx(outImg, &value_min, &value_max, 0, 0);
    for (int l = 0; l < 256; l++){
        float graydivide = float(l)/255 ;
        for (int i = 0; i < outImg.rows; ++i) {
            for (int j = 0; j < outImg.cols; ++j) {
                if(outImg.at<float>(i,j) == float(l)){
                    outImg.at<float>(i,j) = float(255 * pow(graydivide, (1 - cdf_w[l])));
                }
            }
        }
    }
    outImg.convertTo(outImg, CV_8UC1);
}

// Average.
void GetBscanImage(std::vector<cv::Mat>& vecInImgs, cv::Mat& outImg, cv::Mat& refImg){
    Mat fltRef;
    medianBlur(refImg, fltRef, 7);
//    fltRef.convertTo(fltRef, CV_32FC1);
    Mat aveImg;
    aveImg = Mat::zeros(refImg.size(), CV_32FC1);
    refImg.convertTo(refImg, CV_32FC1);
    int aveNum = 0; // keep the number of images averaged.
    for (auto& img : vecInImgs) {
        if(RemoveGhostImage(img, fltRef)){
            Mat dst;
            img.convertTo(dst, CV_32FC1);
            aveImg += dst;
            aveNum++;
        }
    }
    aveImg = aveImg + refImg;
    aveImg /= (aveNum + 1); // aveNum + 1: images after judged and the refImg
    cout << "the number of the images averaged: " << aveNum + 1 << endl;
    aveImg.convertTo(outImg, CV_8UC1);
    imwrite("Bscan.png",outImg);
    //EnhanceBscan(outImg,outImg);
    EnhanceBscanArea(outImg,outImg);
    imwrite("Bscan_enhance.png",outImg);
}

void SignalScore(cv::Mat& inImg){
    double * col_max = new double[inImg.cols];
    for (int i = 0; i < inImg.cols; i++){
        Mat colmat = inImg.col(i).clone();
        minMaxIdx(colmat, NULL, &col_max[i], NULL, NULL);
    }
    double maxsum = 0;
    for (int i = 0; i < inImg.cols; i++){
        maxsum += col_max[i];
    }
    cout << maxsum <<  endl;
    int score = 0;
    if (maxsum < 114 * inImg.cols) score = 1;
    else if (maxsum > 235 * inImg.cols) score = 10;
    else score = (maxsum - 100 * inImg.cols) / (13.5 * inImg.cols);
    cout << score << endl;
    string text = "Signal Strength: " + to_string(score) + "/10";
    putText(inImg, text, Point(650, 50), FONT_HERSHEY_DUPLEX , 0.7, CV_RGB(255,255,255));
    // imshow("img_score", inImg);
    // waitKey(100);
}

int main(){
    const char* s;
    FILE* fp;
    unsigned short * pixdata = new unsigned short[100000 * 2048];
    FFT * filter;
    filter = new FFT;
    Mat_Data * matdata = Mat_Data::GetInstance();

    // read the raw_data
    // s = "rawdata3.txt";
    s = "data1.txt";
    fp = fopen(s,"rb");
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    std::cout << "text size : "<< size << std::endl;
    fseek(fp, 0, SEEK_SET);
    cout << fread(pixdata, sizeof(unsigned short), size / 2, fp) / 2048 << endl;
    matdata->slice = size / (2 * 2048 * matdata->height); std::cout << "slice : " << matdata->slice << std::endl;
    fclose(fp);
    matdata->Img.resize(matdata->slice);
    for (int i = 0; i < matdata->slice; i++) {
        matdata->Img[i] = cv::Mat(matdata->height, 2048, CV_16UC1);
    }
    for (int j = 0; j < matdata->slice; j++) {
        for (int i = 0; i < matdata->height; i++) {
            transpose(cv::Mat(std::vector<unsigned short>(pixdata + i * 2048 + j * matdata->height * 2048, pixdata + (i + 1) * 2048 + j * matdata->height * 2048)), matdata->Img[j].row(i));
        }
    }

    // the parameter of the wasatch(光谱仪)
    double *wave_pix = new double[4];
//    wave_pix[0] = 7.981274095576825e+02; wave_pix[1] = 0.045991067876607; wave_pix[2] = -3.245203344423020e-06; wave_pix[3] = 1.877515501383500e-10;
//    wave_pix[0] = 7.972205933127757e+02; wave_pix[1] = 0.046069702407356; wave_pix[2] = -3.197724352032588e-06; wave_pix[3] = 1.490060930243047e-10; //for data2.txt data3.txt
    wave_pix[0]=8.00608E+02;wave_pix[1]= 4.13653E-02;wave_pix[2]= -1.23689E-06;wave_pix[3]=-5.40202E-11; //for data1.txt
    filter->SetPara(wave_pix[0], wave_pix[1], wave_pix[2], wave_pix[3]);

    //reconstruction and generate the 20 images(1000*1024)
    Mat res_tailored, res;
    std::vector<cv::Mat> vecImgs;
    double gamma_para = 1.05;
    for(int k = 0; k < matdata->slice; k++){
        filter->GetOCTDFTImg_CPU(matdata->Img[k], res);
        res_tailored = res(Rect(0,0,matdata->height,1000));
        normalize(res_tailored, res_tailored, 0, 255, cv::NORM_MINMAX);

        // gamma correction
        for (int i = 0; i < res_tailored.rows; i++) {
            float * p = res_tailored.ptr<float>(i);
            for (int j = 0; j < res_tailored.cols; j++) {
                p[j] = pow(p[j], gamma_para);
            }
        }
        cv::Mat fftMat, filmat;
//        res_tailored.convertTo(fftMat, CV_8UC1);

        /// filter
        res_tailored.convertTo(filmat, CV_8UC1);
        WienerFilter(filmat, fftMat, Size(5, 5));
//        imshow("img_preview", fftMat);
//        waitKey(100);
//        string Img_Name = "img"+to_string(k)+".jpg";
//        imwrite(Img_Name,fftMat);
        if (!fftMat.empty())
            vecImgs.push_back(fftMat);

        Mat fftmat_score = fftMat.clone();
        SignalScore(fftmat_score);
        string Img_Name = "img"+to_string(k)+".jpg";
        imwrite(Img_Name,fftmat_score);
    }


//    std::vector<cv::Mat> vecImgs;
//    vecImgs.clear();
//    string path = "/Users/adam_lau/Downloads/20191121_164222";//此处替换为自己的图片序列
//    vector<String> src_name;
//    glob(path,src_name, false);
//    cout << src_name.size() << endl;
//
//    for (int i = 0; i < src_name.size(); ++i) {
//        cout << src_name[i] << endl;
//        Mat image = imread(src_name[i], 0);
//        if (!image.empty())
//            vecImgs.push_back(image);
//    }

    // find the base_image and the images to be registrated
    Mat baseImg;
    std::vector<cv::Mat> vecImgs_toberegistrated;
    FindBaseImage(vecImgs,baseImg,vecImgs_toberegistrated);

    // Alignment
    std::vector<cv::Mat> vecImgs_registrated;
    std::vector<double> similarity;

    ImageRegistration_s(vecImgs_toberegistrated, baseImg, vecImgs_registrated, similarity);
//    ImageRegistration(vecImgs_toberegistrated, baseImg, vecImgs_registrated, similarity);
////    Mat outmatregistered;
////    image2register(vecImgs_toberegistrated[5], baseImg, outmatregistered);

    // Average and enhance
    Mat Bscan;
    GetBscanImage(vecImgs_registrated, Bscan, baseImg);
    Mat im_color;
    applyColorMap(Bscan, im_color, COLORMAP_JET);
    imwrite("Bscan_color.png", im_color);
    imshow("Bscan_final",im_color);
    waitKey();

    cout << "finished" << endl;
}