//
// Created by raviprabhashankar on 15.10.20.
//

#include <iomanip>
#include <cstdlib>
#include <limits>
#include <random>
#include <mpi/mpi.h>
#include "parameter_analysis.h"
#include "important_algorithms.h"


//Structure variables for the parameters
struct parameters dependent_param, independent_param;
//========================================================================


//File to initialize the independent parameter's values
std::string ind_param_filename;
//========================================================================


//Current Analysis Output(Dataset) filename
std::string analysis_output_filename;
std::string temp_analysis_output_filename;
//========================================================================


//Average Lower Resolution Gaussian Kernel size
int low_resolution_gaussian_kernel_size;
//========================================================================


void setAnalysisEnvironment(struct img_params in_img_features){
    /* ================== Presenting the list of the dependent and independent variables =================
    std::cout<<"Current version has the following dependent and independent parameters available for analysis: "<<std::endl;
    std::cout<<"\nList of dependent parameters: "<<std::endl \
             <<"\t1) Average deviations in Center of Mass methodology: dxdy_centerofmass_avg"<<std::endl;
    std::cout<<"\nList of independent parameters: "<<std::endl \
             <<"\t1) Gaussian Kernel size: gb_kernel_x or gb_kernel_y"<<std::endl \
             <<"\t2) Noise upper limit: noise_upper_limit"<<std::endl;
     */
    std::string file_partial_name;

    /* ================== Scanning the dependent variable and setting the corresponding structure ================= */
    if(dependent_param.index == 1){
        dependent_param.param_name  = "dxdy_centerofmass_avg";
    } else{
        std::cout<<__FILE__<<"("<<__LINE__<<"): " \
                 <<"Wrong index value chosen for the dependent variable. Only values in the list above is permissible. "<<std::endl \
                 <<"Terminating program due to incorrect values"<<std::endl;
        exit(EXIT_FAILURE);
    }


    /* ================== Scanning the independent variable and setting the corresponding structure ================= */
    if(independent_param.index == 1){
        independent_param.param_name    = "gb_kernel_size";
        ind_param_filename              = "./Parameters_data_files/independent_params_gaussiankernel.dat";
        file_partial_name               = "_nup" + std::to_string((int)(percent_max_noise*100))
                                          + "_rmf" + std::to_string(resolution_reduction_factor)
                                          + "_" + std::to_string(in_img_features.imgwd)
                                          + "_" + std::to_string(in_img_features.imgbitsize) + "-bit.txt";
    } else if(independent_param.index == 2){
        independent_param.param_name    = "noise_upper_limit";
        ind_param_filename              = "./Parameters_data_files/independent_params_noise.dat";
        file_partial_name               = "_gb" + std::to_string(gb_kernel_x)
                                          + "_rmf" + std::to_string(resolution_reduction_factor)
                                          + "_" + std::to_string(in_img_features.imgwd)
                                          + "_" + std::to_string(in_img_features.imgbitsize) + "-bit.txt";
    } else{
        std::cout<<__FILE__<<"("<<__LINE__<<"): " \
                 <<"Wrong index value chosen for the independent variable. Only values in the list above is permissible. "<<std::endl \
                 <<"Terminating program due to incorrect values";
        exit(EXIT_FAILURE);
    }

    //Setting the filename for the current run
    analysis_output_filename        = dependent_param.param_name + "_vs_" + independent_param.param_name + file_partial_name;
    temp_analysis_output_filename   = "tmp.current_analysis_run.txt";
}



void init_analysisOutputFile(struct parameters ind_param, struct parameters dep_param){
    //Opening the parsed file and checking for file correctness;
    //CAUTION: file must be opened in std::iostream::trunc for appending the contents into the file
    std::fstream outputfile(temp_analysis_output_filename, std::iostream::out | std::iostream::trunc);
    if(!outputfile){
        std::cerr<<"Error in opening the analysis output file"<<std::endl;
        exit(-1);
    }

    //Writing the independent and dependent parameters values as comments into the file
    outputfile<<"# Parameters Values"<<std::endl \
              <<"#"<<std::setw(40)<<"Lower Bound pixel percent = "<<lower_range_percent<<std::endl \
              <<"#"<<std::setw(40)<<"Upper Bound pixel percent = "<<upper_range_percent<<std::endl \
              <<"#"<<std::setw(40)<<"Resolution Modulation Factor = "<<resolution_reduction_factor<<std::endl;
    if(ind_param.index == 1){
        outputfile<<"#"<<std::setw(40)<<"Noise Upper Bound Percent = "<<percent_max_noise<<std::endl;
    }else if(ind_param.index == 2){
        outputfile<<"#"<<std::setw(40)<<"Gaussian Kernel Size = "<<gb_kernel_x<<"x"<<gb_kernel_y<<std::endl<<std::endl;
    }

    outputfile<<std::endl<<"#"<<independent_param.param_name<<"\t|\t"<<dependent_param.param_name<<std::endl;
    outputfile.close();
}






template <typename T>
void executeAnalysis(cv::Mat input_img, struct img_params in_img_features, int rank, int size, T identifier){
    int first_val, final_val, step_size;
    int data[3];                            //for scanning first_val, final_val and step_size
    double dependent_par_val;               //for saving the analysis return value for the current analysis run

    //Opening the initialization file for the independent parameter values, namely, first_val, final_val and step_size
    std::ifstream parameters_file;
    parameters_file.open(ind_param_filename);
    if(parameters_file.fail()){
        std::cerr<<"Error report from rank: "<<rank<<" file "<<ind_param_filename<<" could not be found in the current path!"<<std::endl;
        exit(1);
    }else {
        if(rank == 0){
            std::cout<<"\t"<<__FILE__<<"("<<__LINE__<<"): File for independent parameters initialization read successfully!"<<std::endl;
        }
    }

    //Scanning of the file to fill in the values in an array
    std::string line;
    int num_values = 0;
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

    //Closing the parameter read file
    parameters_file.close();

    //Assigning values to the actual data (is not really necessary but for the readability of the code)
    first_val = data[0];
    final_val = data[1];
    step_size = data[2];


    //Opening the parameter output file and Terminal prompt
    std::ofstream output_file;
    if(rank == 0){
        output_file.open(temp_analysis_output_filename, std::iostream::out | std::iostream::app);
        std::cout<<"\tFrom "<<__FILE__<<"["<<__FUNCTION__<<"("<<__LINE__<<")]:: ANALYSIS OUTPUT: "<<std::endl;
    }


    //Actual initiation of the analysis
    if(independent_param.index == 1){       //when independent parameter is gaussian blur kernel
        for(int index = first_val; index<=final_val; index = index + step_size){
            gb_kernel_x     = index;
            gb_kernel_y     = index;
            setGBparameters();

            if(img_category == 0){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._i_value, identifier);
            }else if(img_category == 2){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._i_value, identifier);
            }else if(img_category == 5){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._f_value, identifier);
            }

            if(rank == 0){
                output_file<<low_resolution_gaussian_kernel_size<<"\t\t"<<dependent_par_val<<std::endl;
                std::cout<<"\t\tFor "<<independent_param.param_name<<": "<<gb_kernel_x \
                         <<"\t Avg. low resol. GB kernel: "<<low_resolution_gaussian_kernel_size \
						 <<"\t Discretization error: "<<discretization_error \
						 <<"\t "<<dependent_param.param_name<<": "<<dxdy_centerofmass_avg \
                         <<"\t Total error: "<<total_error<<std::endl;
            }
        }
    } else if(independent_param.index == 2){    //when independent parameter is the noise upper bound percentage
        for(int index = first_val; index<=final_val; index = index + step_size){
            if(index == 0){
                noise_present       = 0;
                percent_max_noise   = 0;
            } else {
                noise_present       = 1;
                percent_max_noise   = index;
                setNoiseParameters();
            }


            if(img_category == 0){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._i_value, identifier);
            }else if(img_category == 2){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._i_value, identifier);
            }else if(img_category == 5){
                dependent_par_val = dxdy_centerofmass_avg_analysis(input_img, in_img_features, size, max_pixel_val._f_value, identifier);
            }

            if(rank == 0){
                output_file<<percent_max_noise<<"\t\t"<<dependent_par_val<<std::endl;
                std::cout<<"\t\tFor "<<independent_param.param_name<<": "<<percent_max_noise \
                         <<"\t Avg. low resol. GB kernel: "<<low_resolution_gaussian_kernel_size \
						 <<"\t Discretization error: "<<discretization_error \
						 <<"\t "<<dependent_param.param_name<<": "<<dxdy_centerofmass_avg \
                         <<"\t Total error: "<<total_error<<std::endl;
						 
            }
        }
    }

    if(rank == 0){
        output_file.close();
    }
}
//Explicit instantiation of the template methods to avoid compilation error
template void executeAnalysis<uchar>(cv::Mat input_img, struct img_params in_img_features, int rank, int size, uchar identifier);
template void executeAnalysis<ushort>(cv::Mat input_img, struct img_params in_img_features, int rank, int size, ushort identifier);
template void executeAnalysis<float>(cv::Mat input_img, struct img_params in_img_features, int rank, int size, float identifier);



/* Method for the analysis of the deviations in the Center of Mass methodology:
 * This method is similar to the run_simulation.cpp::generate_frames method with the exception that it doesn't work on
 * image frames but only the sampled coordinates and determines the deviations in the COM methodology
 */
template<typename D, typename T>
double dxdy_centerofmass_avg_analysis(  cv::Mat input_img, struct img_params in_img_features,
                                        int size, D max_pixel_value, T identifier) {

    /* ================== Setting Dimensions and other image parameters ====================== */
    //Setting parameters value for padded high resolution image
    struct img_params pd_img_features = pd_img_data(in_img_features);


    //Dimension for low resolution smoothed output
    int low_resolution_dim_x        = in_img_features.imgwd/resolution_reduction_factor;
    int low_resolution_dim_y        = in_img_features.imght/resolution_reduction_factor;

    //Relation: PADDED_INPUT_IMAGE(pad length, resolution reduction factor) -> PADDED_LOW_RESOLUTION_OUTPUT_IMAGE(pad length)
    //NOTE: The output image for the current code is a lower resolution image based on the reduction factor	===========
    //This part has now been shifted to init_params::setGBparameters() function


    //Low resolution padded image dimension (edited in January 2021 to add comment)
    int pd_low_resolution_dim_y = low_resolution_dim_y + 2*pad_length_low_resolution_y;
    int pd_low_resolution_dim_x = low_resolution_dim_x + 2*pad_length_low_resolution_x;

    //Averaged low resolution gaussian kernel size
    double low_resolution_gb_kernel_sum     = 0.0;
    int criterion_success_counter           = 0;

    //Declaration and assigning values to patch image attributes
    struct img_params patch_img_features;
    patch_img_features.imgwd = gb_kernel_x;
    patch_img_features.imght = gb_kernel_y;
    patch_img_features.imgtyp = input_img.type();
    patch_img_features.imgchannel = input_img.channels();
    if(patch_img_features.imgtyp == CV_8U) {
        patch_img_features.imgbitsize = 8;
    } else if(patch_img_features.imgtyp == CV_16U) {
        patch_img_features.imgbitsize = 16;
    } else if(patch_img_features.imgtyp == CV_32F) {
        patch_img_features.imgbitsize = 32;
    }

    //Low resolution padded image dimension
    double centroid_x, centroid_y, noise_centroid_x, noise_centroid_y;
    double norm_x, norm_y, norm_low_resol_x, norm_low_resol_y, norm_noise_low_resol_x, norm_noise_low_resol_y;


    //Moments
    cv::Moments without_noise, with_noise;


    /* ======== Setting temporary small images like ROIs or GB patches used during the sampling ========== */
    //Patch Images (for computation of Gaussian Blur)
    cv::Mat patch_img = createImage(patch_img_features);    //b.r) Patch Image for High Resolution padded GB (smoothed) output
    cv::Mat low_resolution_patch_img;                       //d.r) Patch Image for Low Resolution padded GB (smoothed) output


    //Noise patch for the lower resolution image
    cv::Mat low_resolution_snp_patch;                       //d.n) noise patch for Low Resolution patch image


    int sampled_row, sampled_col;               //sampled pixel locations on input image
    int global_row, global_col;                 //for determining the coordinates of sampled pixel location based on the new padding scheme
    int sampler_count = 0;                      //counter for while loop
    int COM_invalid_position_flag;              //Flag to check when upscaling COM coordinates overshoot the image dimension
    int COM_invalid_position_counter = 0;       //Counter to track the number of times upscaling COM coordinate overshoots the image dimension
    double distance;

    //Random Number Generation variables
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution_coordinate(0, in_img_features.imght-1);
    std::uniform_real_distribution<double> distribution_rand_num(0, 1);


    /* (a) Array to map the higher resolution GB patch image coordinate to the lower resolution GB patch image coordinate
     * (b) Variables for the local (lower resolution) GB patch image (calculation dependent on the array mentioned in (a) )
     */
    int local_coordinates[4];      // 0 = min_row, 1 = min_col, 2 = max_row, 3 = max_col
    int gb_lower_row, gb_lower_col;
    D pixel_val;

    //Initializing the analysis parameter and the errors
	discretization_error	= 0.0;
	dxdy_centerofmass_avg 	= 0.0;
	total_error				= 0.0;


    while (sampler_count < max_sampler_count_per_proc) {
        //Sampling points on the input image
        sampled_row = distribution_coordinate(gen);
        sampled_col = distribution_coordinate(gen);

        //Global location of the sampled pixel (based on the padded image)
        global_row = sampled_row - halfgaussian_y + pad_length_y;
        global_col = sampled_col - halfgaussian_x + pad_length_x;

        //Assigning the value to the current random number and the pixel value at location (x,y)
        pixel_val = input_img.at<T>(sampled_row, sampled_col);


        if (pixel_val/(double)max_pixel_value >= distribution_rand_num(gen)) {
            //Criterion success counter: for calculation of the average low resolution gaussian kernel size
            criterion_success_counter++;

            //Resetting the upscale coordinate overshoot flag
            COM_invalid_position_flag = 0;
			
			//calculating the normalized coordinates with respect to the high resolution image
            norm_x = (double)(global_col + 1)/pd_img_features.imgwd;
            norm_y = (double)(global_row + 1)/pd_img_features.imght;

            //Processing the patch image
            patch_img = createImage(patch_img_features);
            patch_img.at<T>(halfgaussian_y, halfgaussian_x) = pixel_val;
            cv::GaussianBlur(patch_img, patch_img, cv::Size(gb_kernel_x, gb_kernel_y), 0, 0);


            /* ============= Processing of Lower Resolution Patch Image ===================== */
            //The methods that calculates the lower resolution patch image, sets the local coordinates array and the dimension
            //for the lower resolution patch image
            if(img_category == 0){
                low_resolution_patch_img = coordinate_map_lowresolGB(patch_img,
                                                                     pad_length_low_resolution_x, pad_length_low_resolution_y,
                                                                     global_row, global_col, local_coordinates,
                                                                     gb_lower_row, gb_lower_col, uchar_identifier);
            } else if(img_category == 2){
                low_resolution_patch_img = coordinate_map_lowresolGB(patch_img,
                                                                     pad_length_low_resolution_x, pad_length_low_resolution_y,
                                                                     global_row, global_col, local_coordinates,
                                                                     gb_lower_row, gb_lower_col, ushort_identifier);
            } else if(img_category == 5){
                low_resolution_patch_img = coordinate_map_lowresolGB(patch_img,
                                                                     pad_length_low_resolution_x, pad_length_low_resolution_y,
                                                                     global_row, global_col, local_coordinates,
                                                                     gb_lower_row, gb_lower_col, float_identifier);
            }

            //Computation related to average gaussian kernel size
            low_resolution_gb_kernel_sum += std::max(gb_lower_row, gb_lower_col);

            //Centroiding WITHOUT NOISE on the low resolution patch
            //Assuming openCV convention of origin to be on top-left:
            //m10 is associated with x-coordinate (column), and m01 associated with y-coordinate (row)
            without_noise = cv::moments(low_resolution_patch_img);
            centroid_x = (double)without_noise.m10/without_noise.m00;
            centroid_y = (double)without_noise.m01/without_noise.m00;
			
			//Normalizing coordinates of centroid without noise with respect to low resolution image
            norm_low_resol_x = (centroid_x + local_coordinates[1] + 1)/pd_low_resolution_dim_x;
            norm_low_resol_y = (centroid_y + local_coordinates[0] + 1)/pd_low_resolution_dim_y;


            /* ============== Processing for the salt and pepper noise patch =============== */
            if(noise_present == 1) {
                //Generate salt and pepper noise of the size of lower resolution image
                if (img_category == 0) {
                    low_resolution_snp_patch = cv::Mat::zeros(gb_lower_row, gb_lower_col, CV_8U);
                    cv::randu(low_resolution_snp_patch, lower_noise_limit._i_noise, upper_noise_limit._i_noise);
                } else if (img_category == 2) {
                    low_resolution_snp_patch = cv::Mat::zeros(gb_lower_row, gb_lower_col, CV_16U);
                    cv::randu(low_resolution_snp_patch, lower_noise_limit._i_noise, upper_noise_limit._i_noise);
                } else if (img_category == 5) {
                    low_resolution_snp_patch = cv::Mat::zeros(gb_lower_row, gb_lower_col, CV_32F);
                    cv::randu(low_resolution_snp_patch, lower_noise_limit._f_noise, upper_noise_limit._f_noise);
                }


                //Addition of noise image to the lower resolution patch image
                cv::add(low_resolution_patch_img, low_resolution_snp_patch, low_resolution_patch_img, cv::noArray(), -1);
            }


            /* ====== Processing of the center of mass for low resolution GB patch after noise addition ====== */
            //Assuming openCV convention of origin to be on top-left:
            //m10 is associated with x-coordinate (column), and m01 associated with y-coordinate (row)
            with_noise = cv::moments(low_resolution_patch_img);
            noise_centroid_x = (double)with_noise.m10/with_noise.m00;
            noise_centroid_y = (double)with_noise.m01/with_noise.m00;


			//Normalized coordinates of centroid with noise with respect to the low resolution image
            norm_noise_low_resol_x = (noise_centroid_x + local_coordinates[1] + 1)/pd_low_resolution_dim_x;
            norm_noise_low_resol_y = (noise_centroid_y + local_coordinates[0] + 1)/pd_low_resolution_dim_y;


			//Total deviation in centroid
			distance = (norm_noise_low_resol_x - norm_x)*(norm_noise_low_resol_x - norm_x);
            distance += (norm_noise_low_resol_y - norm_y)*(norm_noise_low_resol_y - norm_y);
            distance = sqrt(distance);
            total_error += distance;

            //Releasing temporary data in order to prevent memory leakage and prevent eating away of the entire RAM during the simulation
            patch_img.release();
            low_resolution_patch_img.release();
            low_resolution_snp_patch.release();
        }

        sampler_count++;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* =========== Processing of the deviations in Centroiding Method =========== */
	double temp_discrete_error 					= discretization_error;
	double temp_dxdy                            = dxdy_centerofmass_avg;
	double temp_total_error 					= total_error;
   
    double temp_low_resolution_gb_kernel_sum    = low_resolution_gb_kernel_sum;
    int temp_criterion_success_counter          = criterion_success_counter;
    int temp_invalid_counter                    = COM_invalid_position_counter;
	
	//MPI_Allreduce(&temp_discrete_error, &discretization_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	//MPI_Allreduce(&temp_dxdy, &dxdy_centerofmass_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&temp_total_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Allreduce(&temp_low_resolution_gb_kernel_sum, &low_resolution_gb_kernel_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&temp_criterion_success_counter, &criterion_success_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&temp_invalid_counter, &COM_invalid_position_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //dxdy_centerofmass_avg = sqrt(dxdy_centerofmass_avg/(criterion_success_counter - COM_invalid_position_counter));
    //dxdy_centerofmass_avg = dxdy_centerofmass_avg/(criterion_success_counter - COM_invalid_position_counter);
	//discretization_error 	= (discretization_error*low_resolution_dim)/max_sampler_count;
	//dxdy_centerofmass_avg 	= (dxdy_centerofmass_avg*low_resolution_dim)/max_sampler_count;
	total_error 			= (total_error*low_resolution_dim_x)/max_sampler_count;
    

    //Low Resolution Gaussian Kernel Size calculation
    low_resolution_gaussian_kernel_size = std::ceil(low_resolution_gb_kernel_sum/criterion_success_counter);

    return dxdy_centerofmass_avg;
}

//Explicit instantiation of the template methods to avoid compilation error
template double dxdy_centerofmass_avg_analysis<int, uchar>(     cv::Mat input_img, struct img_params in_img_features,
                                                                int size, int max_pixel_value, uchar identifier);
template double dxdy_centerofmass_avg_analysis<int, ushort>(    cv::Mat input_img, struct img_params in_img_features,
                                                                int size, int max_pixel_value, ushort identifier);
template double dxdy_centerofmass_avg_analysis<float, float>(   cv::Mat input_img, struct img_params in_img_features,
                                                                int size, float max_pixel_value, float identifier);


void safecopy_final_output_analysis(){
    std::ifstream source(temp_analysis_output_filename);
    std::ofstream dest(analysis_output_filename);

    dest<<source.rdbuf();
}
