/* Created by raviprabhashankar on 10.10.20.
 *
 * Main Purpose:
 *  - Initialization of Parameters for the simulation
 *  - Checking of the values for constraints
 *
 * RELEVANT DEPENDENCY:
 *  - init_visual.h
 */

#include <iostream>
#include "init_params.h"
#include "parameter_analysis.h"


//Identifiers for template function calls based on image  type, i.e., 8, 16 or 32-bit
uchar uchar_identifier      = 0;
ushort ushort_identifier    = 0;
float float_identifier      = 0.0;
//=============================================================================


//Image parameters
int                 img_category;
int                 num_output_frames;
img_value_limits    max_pixel_val;
//=============================================================================


//Changing input image range
float lower_range_percent;
float upper_range_percent;
//=============================================================================


// Gaussian Blur Parameters
int     gb_kernel_x, gb_kernel_y, gb_sigma,
        halfgaussian_x, halfgaussian_y,
        pad_length_low_resolution_x, pad_length_low_resolution_y,
        pad_length_x, pad_length_y;
//=============================================================================


//Sampling Parameters
int     max_sampler_count;
int     max_sampler_count_per_proc;
int     resolution_reduction_factor;
int     noise_present;
//=============================================================================

//Noise Parameters
float               percent_min_noise, percent_max_noise;
noise_limit_vals    upper_noise_limit, lower_noise_limit;
//=============================================================================


//Analysis Parameters
double discretization_error;
double dxdy_centerofmass_avg;
double total_error;

//=============================================================================

void check_execution(int num_args, char* argv[]){
    if(num_args < 3){
        std::cout<<__FILE__<<"("<<__LINE__<<"): "<<std::endl \
                 <<"Program terminated due to insufficient arguments to execute the program!\n" \
                 <<"ERROR: Check for possibly missing arguments, i.e., input image filename and " \
                 <<"parameters filename"<<std::endl;
        exit(EXIT_FAILURE);
    }
}


void readParameters(std::string filename, int rank){
    std::ifstream parameters_file;
    parameters_file.open(filename);
    if(parameters_file.fail()){
        std::cerr<<"Error report from rank "<<rank<<": "<<filename<<" could not be found in the current path!"<<std::endl;
        exit(1);
    }else{
        if(rank == 0){
            std::cout<<"From "<<__FILE__<<"("<<__LINE__<<"):: File for initialization of parameters being read..."<<std::endl;
        }
    }

    //Checking the number of parameters to be read
    int num_values = 0;
    int dummy_vals;
    std::string line;
    while(!parameters_file.eof()){
        while(std::getline(parameters_file, line)){
            if(line[0] != '#'){
                std::istringstream iss(line);
                while((iss>>dummy_vals)){
                    num_values++;
                }
            }
        }
    }

    if(rank == 0){
        std::cout<<"\tTotal number of parameters to be read: "<<num_values<<std::endl;
    }

    //instantiation of the array for the data
    int data[num_values];
    num_values = 0;

    //Resetting the file pointer to the beginning of the file
    parameters_file.clear();
    parameters_file.seekg(0, std::ios::beg);

    //Actual reading of the parameters
    while(!parameters_file.eof()){
        while(std::getline(parameters_file, line)){
            if(line[0] != '#'){
                std::istringstream iss(line);
                while((iss>>data[num_values])){
                    num_values++;
                }
            }
        }
    }

    //Static allocation of the data to the respective parameters
    lower_range_percent         = data[0];
    upper_range_percent         = data[1];
    gb_kernel_x                 = data[2];
    gb_kernel_y                 = data[3];
    gb_sigma                    = data[4];
    max_sampler_count           = data[5];
    resolution_reduction_factor = data[6];
    noise_present               = data[7];
    percent_min_noise           = data[8];
    percent_max_noise           = data[9];
    num_output_frames           = data[10];
    dependent_param.index       = data[11];
    independent_param.index     = data[12];


    parameters_file.close();

    if(rank == 0){
        std::cout<<"Intialization of program parameters successful. File closed!"<<std::endl;
        std::cout<<std::string(star_separator_count, '*')<<std::endl<<std::endl;
    }
}


cv::Mat openImage(const std::string img_filename){
    cv::Mat image = cv::imread(img_filename, cv::IMREAD_ANYDEPTH);

    if(image.empty()){
        std::cout<<"Couldn't find the image "<<img_filename<<" in the current path."<<std::endl;
        std::cin.get();
        exit(EXIT_FAILURE);
    }

    /*If image category is 2, i.e., 16-bit grayscale image, we normalize the input image to 16-bit size.
     * This is done since by default OpenCV opens 16-bit images as 8-bit images
     */
    double min, max;
    if(image.type() == CV_8U) {
        cv::minMaxIdx(image, &min, &max);
        max_pixel_val._i_value  = (int)max;
    } else if(image.type() == CV_16U){
        cv::minMaxIdx(image, &min, &max);
        max_pixel_val._i_value  = (int)max;
    } else if(image.type() == CV_32F){
        cv::minMaxIdx(image, &min, &max);
        max_pixel_val._f_value = (float)max;
        //std::cout<<"Maximum value = "<<max_pixel_val._f_value<<std::endl;
    }

    return image;
}


void displayImage(const cv::Mat image, const std::string windowname){
    cv::namedWindow(windowname, cv::WINDOW_NORMAL);
    cv::imshow(windowname, image);
    cv::waitKey(0);
    cv::destroyWindow(windowname);
}

struct img_params img_data(const cv::Mat image){
    struct img_params data;
    data.imgwd      = image.rows;
    data.imght      = image.cols;
    data.imgchannel = image.channels();
    data.imgtyp     = image.type();

    if(data.imgtyp == CV_8U){
        img_category    = 0;
        data.imgbitsize = 8;
    } else if(data.imgtyp == CV_16U){
        img_category    = 2;
        data.imgbitsize = 16;
    } else if(data.imgtyp == CV_32F){
        img_category    = 5;
        data.imgbitsize = 32;
    }

    return data;
}

template<typename D, typename T> cv::Mat setImgRange(cv::Mat image, D max_pixel_val, int rank, T identifier){
    lower_range_percent = (float)(lower_range_percent/100);
    upper_range_percent = (float)(upper_range_percent/100);

    D upper_val, lower_val;

    if(img_category == 0 || img_category == 2){
        lower_val = (D)(lower_range_percent*max_pixel_val);
        upper_val = (D)(upper_range_percent*max_pixel_val);

        if(rank == 0){
            std::cout<<"From "<<__FILE__<<"("<<__LINE__<<"):: FOR DEBUGGING: lower and upper pixel values are: "\
                     <<lower_val<<" and "<<upper_val<<std::endl;
        }
    }else if(img_category == 2){
        lower_val = (D)(lower_range_percent*max_pixel_val);
        upper_val = (D)(upper_range_percent*max_pixel_val);

        if(rank == 0){
            std::cout<<"From "<<__FILE__<<"("<<__LINE__<<"):: FOR DEBUGGING: lower and upper pixel values are: "\
                     <<lower_val<<" and "<<upper_val<<std::endl;
        }
    }else if(img_category == 5){
        lower_val = (D)(lower_range_percent*max_pixel_val);
        upper_val = (D)(upper_range_percent*max_pixel_val);

        if(rank == 0){
            std::cout<<"From "<<__FILE__<<"("<<__LINE__<<"):: FOR DEBUGGING: lower and upper pixel values are: "\
                     <<lower_val<<" and "<<upper_val<<std::endl;
        }
    }

    D temp;
    for(int index = 0; index < image.rows; index++){
        for(int jindex = 0; jindex < image.cols; jindex++){
            temp = image.at<T>(index, jindex);
            image.at<T>(index, jindex) = (lower_val + temp*(upper_val - lower_val)/max_pixel_val);
        }
    }
	
	if(rank == 0){
		std::string scaled_input_img = "scaled_input_image.tif";
		cv::imwrite(scaled_input_img, image);
	}

    return image;
}

cv::Mat createImage(struct img_params parameters, int val){
    cv::Mat image(parameters.imght, parameters.imgwd, parameters.imgtyp, cv::Scalar::all(val));
    if(image.empty()){
        std::cout<<"FROM CREATE IMAGE: Couldn't open the image with the given parameters.\n" \
                 <<"Program terminating incorrectly!!!";
        exit(EXIT_FAILURE);
    }

    return image;
}


void setGBparameters(){
    //Half length of the gaussian kernel size in x and y-direction
    halfgaussian_x  = gb_kernel_x/2;
    halfgaussian_y  = gb_kernel_y/2;

    //Padding length on low resolution image
    if(halfgaussian_x % resolution_reduction_factor == 0){
        pad_length_low_resolution_x = halfgaussian_x/resolution_reduction_factor;
    } else {
        pad_length_low_resolution_x = 1 + (int)(halfgaussian_x/resolution_reduction_factor);
    }

    if(halfgaussian_y % resolution_reduction_factor == 0){
        pad_length_low_resolution_y = halfgaussian_y/resolution_reduction_factor;
    } else {
        pad_length_low_resolution_y = 1 + (int)(halfgaussian_y/resolution_reduction_factor);
    }

    //Padding length on high resolution image
    pad_length_x = pad_length_low_resolution_x*resolution_reduction_factor;
    pad_length_y = pad_length_low_resolution_y*resolution_reduction_factor;
}


void setSamplingParameters(cv::Mat image, int num_procs, int rank) {
    if(image.rows%resolution_reduction_factor != 0){
        std::cout<<"Input image dimension must be perfectly divisible by resolution reduction factor"<<std::endl;
        std::cout<<"Current values for input image dimension and resolution reduction factor is"<<image.rows<<" and " \
                 <<resolution_reduction_factor<<"respectively" \
                 <<"\nTerminating program due to incorrect values"<<std::endl;;
        exit(EXIT_FAILURE);
    }

    max_sampler_count_per_proc = max_sampler_count/num_procs;
    if(rank == num_procs - 1){
        max_sampler_count_per_proc += max_sampler_count%num_procs;
    }

}


void setNoiseParameters(){
    percent_min_noise = (float)(percent_min_noise/100);    //minimum percent value of noise (w.r.t. the maximum bit-depth)
    percent_max_noise = (float)(percent_max_noise/100);    //maximum percent value of noise (w.r.t. the maximum bit-depth)

    //Temporary Image and pixel value at the center of the image
    cv::Mat temp_img;
    int dim_temp_img_x = gb_kernel_x + 2*halfgaussian_x;
    int dim_temp_img_y = gb_kernel_y + 2*halfgaussian_y;
    int pixel_val_int;
    float pixel_val_float;

    if(img_category == 0){
        //Pixel val for the center pixel for Gaussian Blur for noise upper limit calculation
        pixel_val_int = (int)(upper_range_percent*max_pixel_val._i_value);

        //Create temporary image for calculation of the maximum noise
        temp_img = cv::Mat(dim_temp_img_y, dim_temp_img_x, CV_8U, cv::Scalar::all(0));
        temp_img.at<uchar>(dim_temp_img_y/2, dim_temp_img_x/2) = pixel_val_int;
        cv::GaussianBlur(temp_img, temp_img, cv::Size(gb_kernel_x, gb_kernel_y), 0, 0);
        temp_img = temp_img(cv::Rect(halfgaussian_x, halfgaussian_y, gb_kernel_x, gb_kernel_y));

        lower_noise_limit._i_noise = percent_min_noise*((int)temp_img.at<uchar>(halfgaussian_y, halfgaussian_x));
        upper_noise_limit._i_noise = percent_max_noise*((int)temp_img.at<uchar>(halfgaussian_y, halfgaussian_x));
    } else if(img_category == 2){
        //Pixel val for the center pixel for Gaussian Blur for noise upper limit calculation
        pixel_val_int = (int)(upper_range_percent*max_pixel_val._i_value);

        //Create temporary image for calculation of the maximum noise
        temp_img = cv::Mat(dim_temp_img_y, dim_temp_img_x, CV_16U, cv::Scalar::all(0));
        temp_img.at<ushort>(dim_temp_img_y/2, dim_temp_img_x/2) = pixel_val_int;
        cv::GaussianBlur(temp_img, temp_img, cv::Size(gb_kernel_x, gb_kernel_y), 0, 0);
        temp_img = temp_img(cv::Rect(halfgaussian_x, halfgaussian_y, gb_kernel_x, gb_kernel_y));

        lower_noise_limit._i_noise = percent_min_noise*((int)temp_img.at<ushort>(halfgaussian_y, halfgaussian_x));
        upper_noise_limit._i_noise = percent_max_noise*((int)temp_img.at<ushort>(halfgaussian_y, halfgaussian_x));
    } else if(img_category == 5){
        //Pixel val for the center pixel for Gaussian Blur for noise upper limit calculation
        pixel_val_float = (float)(upper_range_percent*max_pixel_val._f_value);

        //Create temporary image for calculation of the maximum noise
        temp_img = cv::Mat(dim_temp_img_y, dim_temp_img_x, CV_32F, cv::Scalar::all(0));
        temp_img.at<float>(dim_temp_img_y/2, dim_temp_img_x/2) = pixel_val_float;
        cv::GaussianBlur(temp_img, temp_img, cv::Size(gb_kernel_x, gb_kernel_y), 0, 0);
        temp_img = temp_img(cv::Rect(halfgaussian_x, halfgaussian_y, gb_kernel_x, gb_kernel_y));

        lower_noise_limit._f_noise = percent_min_noise*((float)temp_img.at<float>(halfgaussian_y, halfgaussian_x));
        upper_noise_limit._f_noise = percent_max_noise*((float)temp_img.at<float>(halfgaussian_y, halfgaussian_x));
    }
}


struct img_params pd_img_data(struct img_params parameters){
    struct img_params pd_parameters;
    pd_parameters.imgwd     = parameters.imgwd + 2*pad_length_x;
    pd_parameters.imght     = parameters.imght + 2*pad_length_y;
    pd_parameters.imgtyp    = parameters.imgtyp;
    pd_parameters.imgchannel= parameters.imgchannel;
    pd_parameters.imgbitsize= parameters.imgbitsize;

    return pd_parameters;
}

//Explicit instantiation of template function
template cv::Mat setImgRange<int, uchar>(cv::Mat image, int max_pixel_val, int rank, uchar identifier);
template cv::Mat setImgRange<int, ushort>(cv::Mat image, int max_pixel_val, int rank, ushort identifier);
template cv::Mat setImgRange<int, float>(cv::Mat image, int max_pixel_val, int rank, float identifier);
template cv::Mat setImgRange<float, uchar>(cv::Mat image, float max_pixel_val, int rank, uchar identifier);
template cv::Mat setImgRange<float, ushort>(cv::Mat image, float max_pixel_val, int rank, ushort identifier);
template cv::Mat setImgRange<float, float>(cv::Mat image, float max_pixel_val, int rank, float identifier);
