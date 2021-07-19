/* Created by raviprabhashankar on 10.10.20.
 * FILE CONTENT:
 *  - Function to check arguments' sufficiency and argument correctness
 *  - Functions to instantiate and initialize the global variables with respect to cv::GaussianBlur
 *    method in OpenCV
 *  - Image creation, displaying and recording its attributes with respect to OpenCV functions
 *
 * Relevant file(s):
 *  - init_visual.cpp
 */

#ifndef SIMULATIONIMAGES_PARALLELIZED_INIT_PARAMS_H
#define SIMULATIONIMAGES_PARALLELIZED_INIT_PARAMS_H
#include <iostream>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <ctype.h>
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv2/opencv.hpp>
#include <mpi.h>
#define star_separator_count 100      //For terminal output prompt

struct img_params{
    int imgwd;
    int imght;
    int imgtyp;
    int imgbitsize;
    int imgchannel;
};

struct centroid_algorithm_output{
    int values[4];
};

//Image parameters
extern int      img_category;       /* Category value based on the type of image:
                                     * 0 = uchar    (8-bit unsigned integer; bit depth: 0 to 255; CV_8U)
                                     * 1 = schar    (8-bit signed integer; bit depth: -128 to 127; CV_8S)
                                     * 2 = ushort   (16-bit unsigned integer; bit depth: 0 to 65535; CV_16U)
                                     * 3 = short    (16-bit signed integer; bit depth: -32768 to 32767; CV_16S)
                                     * 4 = int      (32-bit signed integer; bit depth: -65536 to 65537; CV_32S)
                                     * 5 = float    (32-bit floating point number)
                                     */
extern int      num_output_frames;  //Number of final output frames


//identifiers to control the function call depending on the type of image, i.e., 8-bit or 16-bit or 32-bit
extern uchar uchar_identifier;
extern ushort ushort_identifier;
extern float float_identifier;
//=============================================================================


//identifier for data type identification to control template function call
//extern int int_data_identifier;
//extern float float_data_identifier;
//=============================================================================


//maximum pixel value in the input image
union img_value_limits{
    int     _i_value;
    float   _f_value;
};
extern img_value_limits max_pixel_val;  //Used for polymorphism behavior in code: same memory location acts as different data types
//=============================================================================


//Changing input image range
extern float lower_range_percent;
extern float upper_range_percent;
//=============================================================================


// Gaussian Blur Parameters
extern int      gb_kernel_x, gb_kernel_y, gb_sigma,
                halfgaussian_x, halfgaussian_y,
                pad_length_low_resolution_x, pad_length_low_resolution_y,
                pad_length_x, pad_length_y;
//=============================================================================


//Sampling Parameters
extern int      max_sampler_count;
extern int      max_sampler_count_per_proc;     //maximum sampling per process working in parallel
extern int      resolution_reduction_factor;    //ratio (input image resolution/output image resolution)
extern int      noise_present;                  //0 = no noise is present, 1 = noise present in the simulation
//=============================================================================


//Noise Parameters
extern float percent_min_noise, percent_max_noise;

union noise_limit_vals{
    int     _i_noise;
    float   _f_noise;
};
extern noise_limit_vals upper_noise_limit, lower_noise_limit;
//=============================================================================


//Analysis Parameters
extern double discretization_error;
extern double dxdy_centerofmass_avg;
extern double total_error; //this includes the error due to discretization and the addition of noise

void check_execution(int, char* argv[]);                //Check the parameters correctness for program execution
void readParameters(std::string, int);
cv::Mat openImage(const std::string);                   //opening the image file to read the input data
void displayImage(const cv::Mat, const std::string);    //GUI display of image
struct img_params img_data(const cv:: Mat);             //Obtaining input image attributes
template<typename D, typename T>
    cv::Mat setImgRange(cv::Mat, D, int, T);                 //Settng image range
cv::Mat createImage(struct img_params, int val = 0);    //Creating an image in OpenCV; 'val' indicates the initialization value for the pixel
void setGBparameters();                                 //Setting Gaussian Blur parameters
void setSamplingParameters(cv::Mat, int, int);          //Setting Sampling parameters
void setNoiseParameters();                              //Setting Noise parameters
struct img_params pd_img_data(struct img_params);       //Obtaining padded image attributes; dependent on input image attributes


#endif //SIMULATIONIMAGES_PARALLELIZED_INIT_VISUAL_H
