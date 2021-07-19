//
// Created by raviprabhashankar on 07.03.21.
//

#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
#include "init_params.h"
#include "important_algorithms.h"

/* ------------------------------------ Coordinate Mapping Algorithm ------------------------------------ */
template<typename T>
cv::Mat coordinate_map_lowresolGB(  cv::Mat patch_img, int pad_length_low_resolution_x, int pad_length_low_resolution_y,
                                    int global_row, int global_col,
                                    int* local_coordinates, int& gb_lower_row, int& gb_lower_col,
                                    T identifier){
    //Creating output patch (that returns from this method)
    cv::Mat output_patch;

    /* =============== Processing of min_row and max_row using the new mapping logic =============== */
    if(pad_length_low_resolution_y == 1) {
        // Minimum row logic:: local_coordinates[0]
        if (global_row < pad_length_y) {
            local_coordinates[0] = (pad_length_y - 1 - global_row) / resolution_reduction_factor;
        } else {
            local_coordinates[0] = pad_length_low_resolution_y + (global_row - pad_length_y) / resolution_reduction_factor;
        }

        //Maximum row logic:: local_coordinates[2]
        //the maximum global row value will always be greater than or equal to pad length, so no if-else condition here
        local_coordinates[2] = pad_length_low_resolution_y + ((global_row + gb_kernel_y - 1) - pad_length_y) / resolution_reduction_factor;
    } else {
        // Minimum row logic:: local_coordinates[0]
        if(global_row < pad_length_y){
            local_coordinates[0] = (pad_length_low_resolution_y  - 1) - ((pad_length_y - 1 - global_row)/resolution_reduction_factor);
        } else {
            local_coordinates[0] = pad_length_low_resolution_y + (global_row - pad_length_y)/resolution_reduction_factor;
        }

        //Maximum row logic:: local_coordinates[2]
        //the maximum global row value will always be greater than or equal to pad length, so no if-else condition here
        local_coordinates[2] = pad_length_low_resolution_y + ((global_row + gb_kernel_y - 1) - pad_length_y)/resolution_reduction_factor;
    }


    /* =============== Processing of min_col and max_col using the new mapping logic =============== */
    if(pad_length_low_resolution_x == 1) {
        //Minimum col logic:: local_coordinates[1]
        if (global_col < pad_length_x) {
            local_coordinates[1] = (pad_length_x - 1 - global_col) / resolution_reduction_factor;
        } else {
            local_coordinates[1] = pad_length_low_resolution_x + (global_col - pad_length_x) / resolution_reduction_factor;
        }

        //Maximum col logic:: local_coordinates[3]
        //the maximum global col value will always be greater than or equal to pad length, so no if-else condition here
        local_coordinates[3] = pad_length_low_resolution_x + ((global_col + gb_kernel_x - 1) - pad_length_x) / resolution_reduction_factor;
    } else {
        //Minimum col logic:: local_coordinates[1]
        if(global_col < pad_length_x){
            local_coordinates[1] = (pad_length_low_resolution_x - 1) - ((pad_length_x - 1 - global_col)/resolution_reduction_factor);
        } else {
            local_coordinates[1] = pad_length_low_resolution_x + (global_col - pad_length_x)/resolution_reduction_factor;
        }

        //Maximum col logic:: local_coordinates[3]
        local_coordinates[3] = pad_length_low_resolution_x + ((global_col + gb_kernel_x - 1) - pad_length_x)/resolution_reduction_factor;
    }

    /* ================= Calculation of the dimensions of the lower resolution patch image ================== */
    gb_lower_row          = local_coordinates[2] - local_coordinates[0] + 1;
    gb_lower_col          = local_coordinates[3] - local_coordinates[1] + 1;


    //Initializing the output patch image
    if(img_category == 0){
        output_patch =  cv::Mat(gb_lower_row, gb_lower_col, CV_8U, cv::Scalar::all(0));
    } else if(img_category == 2){
        output_patch =  cv::Mat(gb_lower_row, gb_lower_col, CV_16U, cv::Scalar::all(0));
    } else if(img_category == 5){
        output_patch =  cv::Mat(gb_lower_row, gb_lower_col, CV_32F, cv::Scalar::all(0));
    }


    int local_row, local_col;
    //Mapping the pixels of high resolution patch as summed contributions to the pixels on lower resolution patch
    if(pad_length_low_resolution_y == 1 && pad_length_low_resolution_x == 1){
        for(int index = 0; index < gb_kernel_y; index++) {
            if(global_row + index < pad_length_y){
                local_row = (pad_length_y - 1 - (global_row + index))/resolution_reduction_factor;
            } else {
                local_row = pad_length_low_resolution_y + ((global_row + index) - pad_length_y)/resolution_reduction_factor;
            }

            for (int jindex = 0; jindex < gb_kernel_x; jindex++) {
                if(global_col + jindex < pad_length_x){
                    local_col = (pad_length_x - 1 - (global_col + jindex))/resolution_reduction_factor;
                } else {
                    local_col = pad_length_low_resolution_x + ((global_col + jindex) - pad_length_x)/resolution_reduction_factor;
                }

                output_patch.at<T>(local_row - local_coordinates[0], local_col - local_coordinates[1]) += patch_img.at<T>(index, jindex);
            }
        }
    } else if(pad_length_low_resolution_y == 1 && pad_length_low_resolution_x > 1){
        for(int index = 0; index < gb_kernel_y; index++) {
            if(global_row + index < pad_length_y){
                local_row = (pad_length_y - 1 - (global_row + index))/resolution_reduction_factor;
            } else {
                local_row = pad_length_low_resolution_y + ((global_row + index) - pad_length_y)/resolution_reduction_factor;
            }

            for (int jindex = 0; jindex < gb_kernel_x; jindex++) {
                if(global_col + jindex < pad_length_x){
                    local_col = (pad_length_low_resolution_x - 1) - ((pad_length_x - 1 - (global_col + jindex))/resolution_reduction_factor);
                } else {
                    local_col = pad_length_low_resolution_x + ((global_col + jindex) - pad_length_x)/resolution_reduction_factor;
                }

                output_patch.at<T>(local_row - local_coordinates[0], local_col - local_coordinates[1]) += patch_img.at<T>(index, jindex);
            }
        }
    } else if(pad_length_low_resolution_y > 1 && pad_length_low_resolution_x == 1){
        for(int index = 0; index < gb_kernel_y; index++) {
            if(global_row + index < pad_length_y){
                local_row = (pad_length_low_resolution_y - 1) - ((pad_length_y - 1 - (global_row + index))/resolution_reduction_factor);
            } else {
                local_row = pad_length_low_resolution_y + ((global_row + index) - pad_length_y)/resolution_reduction_factor;
            }

            for (int jindex = 0; jindex < gb_kernel_x; jindex++) {
                if(global_col + jindex < pad_length_x){
                    local_col = (pad_length_x - 1 - (global_col + jindex))/resolution_reduction_factor;
                } else {
                    local_col = pad_length_low_resolution_x + ((global_col + jindex) - pad_length_x)/resolution_reduction_factor;
                }

                output_patch.at<T>(local_row - local_coordinates[0], local_col - local_coordinates[1]) += patch_img.at<T>(index, jindex);
            }
        }
    } else {
        for(int index = 0; index < gb_kernel_y; index++) {
            if(global_row + index < pad_length_y){
                local_row = (pad_length_low_resolution_y - 1) - ((pad_length_y - 1 - (global_row + index))/resolution_reduction_factor);
            } else {
                local_row = pad_length_low_resolution_y + ((global_row + index) - pad_length_y)/resolution_reduction_factor;
            }

            for (int jindex = 0; jindex < gb_kernel_x; jindex++) {
                if(global_col + jindex < pad_length_x){
                    local_col = (pad_length_low_resolution_x - 1) - ((pad_length_x - 1 - (global_col + jindex))/resolution_reduction_factor);
                } else {
                    local_col = pad_length_low_resolution_x + ((global_col + jindex) - pad_length_x)/resolution_reduction_factor;
                }

                output_patch.at<T>(local_row - local_coordinates[0], local_col - local_coordinates[1]) += patch_img.at<T>(index, jindex);
            }
        }
    }

    /*std::cout<<__FILE__<<"("<<__LINE__<<"):: Recheck: global row-col: "<<global_row<<", "<<global_col;
    std::cout<<"\tCorresponding Local coordinates: ("<<local_coordinates[0]<<","<<local_coordinates[1]<<"); ("<<local_coordinates[2]<<", "<<local_coordinates[3]<<")";
    std::cout<<"low resolution patch image dimension: "<<output_patch.rows<<",\t"<<output_patch.cols<<std::endl; //Probes*/
    return output_patch;
}
//Explicit instantiation of the above associated template function
template cv::Mat coordinate_map_lowresolGB<uchar>(cv::Mat patch_img,
                                                  int pad_length_low_resolution_x,
                                                  int pad_length_low_resolution_y,
                                                  int global_row, int global_col,
                                                  int* local_coordinates, int& gb_lower_row, int& gb_lower_col,
                                                  uchar identifier);
template cv::Mat coordinate_map_lowresolGB<ushort>(cv::Mat patch_img,
                                                   int pad_length_low_resolution_x,
                                                   int pad_length_low_resolution_y,
                                                   int global_row, int global_col,
                                                   int* local_coordinates, int& gb_lower_row, int& gb_lower_col,
                                                   ushort identifier);
template cv::Mat coordinate_map_lowresolGB<float>(cv::Mat patch_img,
                                                  int pad_length_low_resolution_x,
                                                  int pad_length_low_resolution_y,
                                                  int global_row, int global_col,
                                                  int* local_coordinates, int& gb_lower_row, int& gb_lower_col,
                                                  float identifier);



/* ------------------------------------ Gaussian/3-point Centroiding Algorithm ------------------------------------ */
template<typename D, typename T>
centroid_algorithm_output gaussian_3_point_centroiding_algorithm(cv::Mat patch, D data_type_identifier, T img_type_identifier){
    D ax, bx, cx, ay, by, cy;       // Pixel values for a, b, c in x and y-direction when image type is unsigned char or short
                                    // bx = by

    double numerator, denominator;                          // Temporary variable for calculation of fraction in gaussian-based centroiding
    double x_cog_frac_pos, y_cog_frac_pos;                  // Fractional position of center of gravity in the pixel on low resolution with highest intensity
    //static int cog_output[4];                               // Actual position of COG (in terms of the pixel within) the resolved highest intensity pixel on low resolution;
                                                            // 0 = row-location of maximum intensity pixel on the patch in patch coordinate reference
                                                            // 1 = col-location of maximum intensity pixel on the patch in patch coordinate reference
                                                            // 2 = row, 3 = col: COG location within the resolved pixel with maximum pixel intensity
    centroid_algorithm_output cog_output;


    cv::Point max_pixel_loc;                                // Maximum pixel location on the patch
    cv::minMaxLoc(patch, NULL, NULL, NULL, &max_pixel_loc, cv::noArray());


    //Values for x-centroiding (ax, bx, and cx correspond to left, center and right w.r.t. the maximum pixel location)
    if(max_pixel_loc.x - 1 < 0){
        ax = 0;
    } else{
        ax = (D)patch.at<T>(max_pixel_loc.y, max_pixel_loc.x - 1);
    }
    bx = (D)patch.at<T>(max_pixel_loc.y, max_pixel_loc.x);
    if(max_pixel_loc.x + 1 >= patch.cols){
        cx = 0;
    } else{
        cx = (D)patch.at<T>(max_pixel_loc.y, max_pixel_loc.x + 1);
    }


    //Values for y-centroiding (ay, by, and cy correspond to top, center and down w.r.t. the maximum pixel location)
    if(max_pixel_loc.y - 1 < 0){
        ay = 0;
    } else{
        ay = (D)patch.at<T>(max_pixel_loc.y - 1, max_pixel_loc.x);
    }
    by = (D)patch.at<T>(max_pixel_loc.y, max_pixel_loc.x);
    if(max_pixel_loc.y + 1 >= patch.rows){
        cy = 0;
    } else{
        cy = (D)patch.at<T>(max_pixel_loc.y + 1, max_pixel_loc.x);
    }

    //Computation of x-centroid
    if((ax == bx && bx == cx) || ax == 0 || cx == 0){ //Apply 3-point COG
        x_cog_frac_pos = ((double)(cx - ax))/(ax + bx + cx);
    } else { //Apply Gaussian based centroiding
        numerator = (double)cx/ax;
        numerator = log(numerator);
        denominator = (double)(bx*bx)/(ax*cx);
        denominator = 2*log(denominator);
        x_cog_frac_pos = numerator/denominator;
    }

    //Computation of y-centroid
    if((ay == by && by == cy) || ay == 0 || cy == 0){ //Apply 3-point centroiding
        y_cog_frac_pos = ((double)(cy - ay))/(ay + by + cy);
    } else { //Apply Gaussian centroiding
        numerator = (double)cy/ay;
        numerator = log(numerator);
        denominator = (double)(by*by)/(ay*cy);
        denominator = 2*log(denominator);
        y_cog_frac_pos = numerator/denominator;
    }
    //std::cout<<"Fraction position of COG (row, col): "<<y_cog_frac_pos<<", "<<x_cog_frac_pos<<std::endl;

    //Finding the COG pixel on the resolved pixel of interest on low resolution
    cog_output.values[0] = max_pixel_loc.y;                                                // Row location of pixel with maximum intensity on low resolution
    cog_output.values[1] = max_pixel_loc.x;                                                // Col location of pixel with maximum intensity on low resolution
    cog_output.values[2] = std::min((int)((y_cog_frac_pos + 0.5)*resolution_reduction_factor), resolution_reduction_factor - 1);      // row location of COG
    cog_output.values[3] = std::min((int)((x_cog_frac_pos + 0.5)*resolution_reduction_factor), resolution_reduction_factor - 1);      // column location of COG

    //std::cout<<"Finishing the COG with Gaussian/3-point centroiding algorithm"<<std::endl;
    //std::cout<<"Patch size:: row = "<<patch.rows<<", cols = "<<patch.cols<<"; Maximum location (row, col): "<<cog_output.values[0]<<", "<<cog_output.values[1] \
    //         <<"; Resolved COG location (row or y, col or x): "<<cog_output.values[2]<<", "<<cog_output.values[3]<<std::endl;
    return cog_output;
}
//Explicit instantiation of the above associated template function
template centroid_algorithm_output gaussian_3_point_centroiding_algorithm<int, uchar>(cv::Mat patch, int data_type_identifier, uchar img_type_identifier);
template centroid_algorithm_output gaussian_3_point_centroiding_algorithm<int, ushort>(cv::Mat patch, int data_type_identifier, ushort img_type_identifier);
template centroid_algorithm_output gaussian_3_point_centroiding_algorithm<float, float>(cv::Mat patch, float data_type_identifier, float img_type_identifier);
