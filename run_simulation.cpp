/* Created by raviprabhashankar on 10.10.20.
 *
 * RELEVANT DEPENDENCY:
 *  - run_simulation.h
 */

#include <iostream>
#include <cmath>
//#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <mpi.h>
#include "run_simulation.h"
#include "init_params.h"
#include "important_algorithms.h"

//Function for coordinate and pixel-value mapping from higher resolution to lower resolution GB patch
//CAUTION: the output local_coordinates has values including the padding w.r.t. lower resolution image

template<typename D, typename T>
cv::Mat* generate_frame(   cv::Mat input_img, struct img_params in_img_features,
                           int rank, int size, D max_pixel_value, T identifier) {

    /* ================== Setting Dimensions and other image parameters ====================== */
    //Setting parameters value for padded high resolution image
    struct img_params pd_img_features 	= pd_img_data(in_img_features);


    //Dimension for low resolution smoothed output
    int low_resolution_dim_x        = in_img_features.imgwd/resolution_reduction_factor;
    int low_resolution_dim_y        = in_img_features.imght/resolution_reduction_factor;

    //Relation: PADDED_INPUT_IMAGE(pad length, resolution reduction factor) -> PADDED_LOW_RESOLUTION_OUTPUT_IMAGE(pad length)
    //NOTE: The output image for the current code is a lower resolution image based on the reduction factor	===========
    //This part has now been shifted to init_params::setGBparameters() function
	
	//Low resolution padded image dimension (edited in January 2021 to add comment)
	int pd_low_resolution_dim_y = low_resolution_dim_y + 2*pad_length_low_resolution_y;
	int pd_low_resolution_dim_x = low_resolution_dim_x + 2*pad_length_low_resolution_x;

    //Declaration and assigning values to patch image attributes
    //Updated on March 09th, 2021 because there was an issue with the application of Gaussian Blur
    struct img_params patch_img_features;
    patch_img_features.imgwd = gb_kernel_x + 2*halfgaussian_x;  //updated
    patch_img_features.imght = gb_kernel_y + 2*halfgaussian_y;  //updated
    patch_img_features.imgtyp = input_img.type();
    patch_img_features.imgchannel = input_img.channels();
    if(patch_img_features.imgtyp == CV_8U) {
        patch_img_features.imgbitsize = 8;
    } else if(patch_img_features.imgtyp == CV_16U) {
        patch_img_features.imgbitsize = 16;
    } else if(patch_img_features.imgtyp == CV_32F) {
        patch_img_features.imgbitsize = 32;
    }


    /* ================== Setting the image frames for all the images to be generated post sampling ================= */
    //Output for the simulation and smoothing images
    cv::Mat summed_sim_img                      = createImage(in_img_features);         		//A) High Resolution ideal simulation output
    cv::Mat summed_smooth_img;                                                  		        //B) High Resolution Gaussian Blur (smoothed) output
    cv::Mat low2high_summed_sim_img  	        = cv::Mat::zeros(in_img_features.imght, in_img_features.imgwd, CV_32F);
																						//C) High Resolution simulation output (post center-of-mass
                                                                                		//   calculation on lower resolution and projected back
                                                                                		//   to high resolution
    cv::Mat low2high_summed_sim_FPN_adjusted    = cv::Mat::zeros(in_img_features.imght, in_img_features.imgwd, CV_32F);
                                                                                        //D) High Resolution simulation output without FPN
                                                                                        //   Calculated using the Gaussian/3-point algorithm
    cv::Mat low_resolution_smooth_img           = cv::Mat::zeros(low_resolution_dim_y, low_resolution_dim_x, CV_32F);
                                                                                		//E) Low Resolution Gaussian Blur (smoothed) output


    //Output for the padded simulation and smoothing images
    //CHANGED OCTOBER 29TH: cv::Mat summed_pd_smooth_img        = createImage(pd_img_features);         			//B.p) High Resolution Padded Gaussian Blur(smoothed) output
	cv::Mat summed_pd_smooth_img        = cv::Mat::zeros(pd_img_features.imght, pd_img_features.imgwd, CV_32F);		//B.p) High Resolution Padded Gaussian Blur(smoothed) output
    cv::Mat low2high_summed_pd_sim_img  = cv::Mat::zeros(pd_img_features.imght, pd_img_features.imgwd, CV_32F);
																				//C.p) High Resolution padded output projected from low resolution centroid outputs
																				//   (post center-of-mass calculation on lower resolution and
                                                                				//   projected back to high resolution
																				//Added and change made in the code on March 03rd, 2021: 
																				//Just scale the padded low resolution
																				//image by multiplying the dimension with RSF
    cv::Mat low2high_summed_pd_sim_FPN_adjusted = cv::Mat::zeros(pd_img_features.imght, pd_img_features.imgwd, CV_32F);
                                                                                //D.p) High Resolution padded output with FPN adjusted
                                                                                //     created by projecting the centroid location calculated using the Gaussian/3-point centroid algorithm
                                                                                //     from low to high resolution
    //CHANGED OCTOBER 29TH: cv::Mat low_resolution_pd_smooth_img   = cv::Mat( low_resolution_dim + 2*pad_length_low_resolution,
    //                                                  					  low_resolution_dim + 2*pad_length_low_resolution,
    //                                                  					  in_img_features.imgtyp, cv::Scalar::all(0));
	cv::Mat low_resolution_pd_smooth_img   = cv::Mat::zeros(pd_low_resolution_dim_y, pd_low_resolution_dim_x, CV_32F);
                                                                                //E.p) Low Resolution Padded Gaussian Blur (smoothed) output

    //Coordinates of centroid with and without noise
    double centroid_x, centroid_y, noise_centroid_x, noise_centroid_y;
	
	//Coordinates of actual centroid on low resolution image, i.e., (location on the mapped patch + local coordinates)
	double low_resol_actual_centroid_x, low_resol_actual_centroid_y;
	
	//Projected coordinates location on high resolution image (location of centroid projected from low resolution to high resolution)
	int projected_x, projected_y;
	
	//Normalized coordinates of interest
    double norm_x, norm_y, norm_low_resol_x, norm_low_resol_y, norm_noise_low_resol_x, norm_noise_low_resol_y;

    //Moments on low resolution image
    cv::Moments without_noise, with_noise;


    /* ======== Setting temporary small images like ROIs or GB patches used during the sampling ========== */
    //Patch Images (for computation of Gaussian Blur)
    cv::Mat patch_img = createImage(patch_img_features);    //b.r) Patch Image for High Resolution padded GB (smoothed) output
	cv::Mat low_resolution_patch_img;                       //d.r) Patch Image for Low Resolution padded GB (smoothed) output
	
    //ROIs on padded image for adding patches
    cv::Mat pd_smooth_ROI;                                  //b.t) ROI for Higher Resolution Padded Gaussian Blur (smoothed) output
    cv::Mat pd_lower_smooth_ROI;                            //d.t) ROI for Lower Resolution Padded Gaussian Blur (smoothed) output

    //Noise patch for the lower resolution image
    cv::Mat low_resolution_snp_patch;                       //d.n) noise patch for Low Resolution patch image
	
	//Temporary patch for the conversion of patch images to 32-bit
	cv::Mat temp_patch_img;

	//Other variables for the simulation
    int sampled_row, sampled_col;               //sampled pixel locations on input image
    int global_row, global_col;                 //for determining the coordinates of sampled pixel location based on the new padding scheme
    int sampler_count = 0;                      //counter for while loop
    //int COM_invalid_position_flag;            //Flag to check when upscaling COM coordinates overshoot the image dimension
    //int COM_invalid_position_counter = 0;     //Counter to track the number of times upscaling COM coordinate overshoots the image dimension
    double distance;


    //Random Number Generation variables
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution_coordinate(0, in_img_features.imght-1);
    std::uniform_real_distribution<double> distribution_rand_num(0, 1);


    /* (a) Array to map the higher resolution GB patch image coordinate to the lower resolution GB patch image coordinate
     * (b) Variables for the local (lower resolution) GB patch image (calculation dependent on the array mentioned in (a) )
     */
    int local_coordinates[4];       // 0 = min_row, 1 = min_col, 2 = max_row, 3 = max_col
    //int* COG_output = new int[4];                // (in patch coordinate reference) 0 = row-location; 1 = col-location: pixel with maximum intensity on the patch
                                    // 2 = row; 3 = column: location of the pixel within the resolved pixel of highest intensity on low resolution
    centroid_algorithm_output COG_output;
    int gb_lower_row, gb_lower_col;
    D pixel_val;
	double check_pixel_val;

    //Initializing the analysis parameter and the errors
	discretization_error	= 0.0;
	dxdy_centerofmass_avg 	= 0.0;
	total_error				= 0.0;

    int criterion_success_counter = 0;

    //std::cout<<"I am here before while"<<std::endl; //&& criterion_success_counter == 0

    while (sampler_count < max_sampler_count_per_proc) {
        //Sampled pixels on the input image
        sampled_row = distribution_coordinate(gen);
        sampled_col = distribution_coordinate(gen);

        //Global location of the sampled pixel (based on the padded image)
        global_row = sampled_row - halfgaussian_y + pad_length_y;
        global_col = sampled_col - halfgaussian_x + pad_length_x;

        //Assigning the value to the current random number and the pixel value at location (x,y)
        pixel_val = input_img.at<T>(sampled_row, sampled_col);  //'at' uses (row, col) convention


        if (pixel_val/(double)max_pixel_value >= distribution_rand_num(gen)) {
            //Probe to check the coordinates of the sampled patch and its pixel intensity
            //std::cout<<__FILE__<<"("<<__LINE__<<"):: Sampler count = " << sampler_count << "\t(row, col, pixel_value):: (" << global_row << "\t" << global_col << "\t" \
            //     << pixel_val << ")\t"; //Probes
            criterion_success_counter++;

            //Resetting the upscale coordinate overshoot flag
            //COM_invalid_position_flag = 0;
			
			//calculating the normalized coordinates with respect to the high resolution image
			norm_x = (double)(global_col + 1)/pd_img_features.imgwd;
			norm_y = (double)(global_row + 1)/pd_img_features.imght;

            //Processing the patch image
            patch_img = createImage(patch_img_features);
            patch_img.at<T>(patch_img_features.imght/2, patch_img_features.imgwd/2) = pixel_val;
            cv::GaussianBlur(patch_img, patch_img, cv::Size(gb_kernel_x, gb_kernel_y), 0);
            patch_img = patch_img(cv::Rect(halfgaussian_x, halfgaussian_y, gb_kernel_x, gb_kernel_y));
			check_pixel_val = cv::sum(patch_img)[0];
			//std::cout<<"patch image sum: "<<check_pixel_val;

            /* ============= Processing of Padded smooth output image ====================== */
            pd_smooth_ROI = summed_pd_smooth_img(cv::Rect(sampled_col, sampled_row, gb_kernel_x, gb_kernel_y));
			
			//Converting the patch_img into 32-bit
			temp_patch_img = patch_img;
			temp_patch_img.convertTo(temp_patch_img, CV_32F);

            //addition of the GB patch image to the padded smooth image
            cv::add(pd_smooth_ROI, temp_patch_img, pd_smooth_ROI, cv::noArray(), -1);

            /* ============= Processing of Simulated output image ============ */
            if (summed_sim_img.at<T>(sampled_row, sampled_col) + (D) 1.0 > max_pixel_value) { //summed_sim_img.at<T>(x, y) + pixel_val > max_pixel_value
                summed_sim_img.at<T>(sampled_row, sampled_col) = max_pixel_value;
            } else {
                summed_sim_img.at<T>(sampled_row, sampled_col) += (D) 1.0;
            }


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

            //Probe to check the correctness of Gaussian Blur application
			/*check_pixel_val = cv::sum(low_resolution_patch_img)[0];
			std::cout<<"\tgb lower row: "<<gb_lower_row<<";\t gb lower col: "<<gb_lower_col<<";\t pixel value: "<<check_pixel_val<<std::endl;*/
	
            //Centroiding WITHOUT NOISE on the low resolution patch
            //Assuming openCV convention of origin to be on top-left:
            //m10 is associated with x-coordinate (column), and m01 associated with y-coordinate (row)
            without_noise = cv::moments(low_resolution_patch_img);
            centroid_x = (double)without_noise.m10/without_noise.m00;
            centroid_y = (double)without_noise.m01/without_noise.m00;
			
			//Normalizing coordinates of centroid without noise with respect to low resolution image
			norm_low_resol_x = (centroid_x + local_coordinates[1] + 1)/pd_low_resolution_dim_x;
			norm_low_resol_y = (centroid_y + local_coordinates[0] + 1)/pd_low_resolution_dim_y;

			//Probe for roughly estimating the projected coordinates without addition of noise and its pixel intensity post application of Pixel Mapping Algorithm
			//std::cout<<"\tProjected Centroid w/o noise: (row, column, pixel): ("<<(int)(norm_low_resol_y*pd_img_features.imght)<<", "<<(int)(norm_low_resol_x*pd_img_features.imgwd) \
			//         <<",\t"<<without_noise.m00<<")"<<std::endl;

            //Probe to check the correctness of the output of Pixel Mapping Algorithm
            /*std::cout<<__FILE__<<"("<<__LINE__<<")::" \
					 <<"min:(row, col) max:(row, col) gb_lower_row gb_lower_col: ("<<local_coordinates[0]<<", "<<local_coordinates[1] \
                     <<")\t("<<local_coordinates[2]<<", "<<local_coordinates[3]<<"),\t"<<gb_lower_row<<",\t"<<gb_lower_col; //Probes
			std::cout<<"\nlow resolution patch image dimension: "<<low_resolution_patch_img.rows<<",\t"<<low_resolution_patch_img.cols; //Probes*/


            /* ============== Processing of Padded Lower Resolution Patch image ============= */
            //CAUTION: for creating the pd_lower_smooth_ROI, the ROI with respect to the padded lower resolution smooth
            //image, one doesn't need to add the lower resolution pad length because local_coordinates takes that into account
            pd_lower_smooth_ROI = low_resolution_pd_smooth_img(cv::Rect(local_coordinates[1],local_coordinates[0],
                                                                        gb_lower_col, gb_lower_row));

			//Converting low_resolution_patch_img to 32-bit
			temp_patch_img = low_resolution_patch_img;
			temp_patch_img.convertTo(temp_patch_img, CV_32F);
			
            //addition of the lower resolution patch image to the lower resolution padded smooth image
            cv::add(pd_lower_smooth_ROI, temp_patch_img, pd_lower_smooth_ROI, cv::noArray(), -1);


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
            noise_centroid_x = (double)(with_noise.m10/with_noise.m00);
            noise_centroid_y = (double)(with_noise.m01/with_noise.m00);


			//Normalized coordinates of centroid with noise with respect to the low resolution image
			norm_noise_low_resol_x = (noise_centroid_x + local_coordinates[1] + 1)/pd_low_resolution_dim_x;
			norm_noise_low_resol_y = (noise_centroid_y + local_coordinates[0] + 1)/pd_low_resolution_dim_y;

			
			//Total deviation in centroid
			distance = (norm_noise_low_resol_x - norm_x)*(norm_noise_low_resol_x - norm_x);
            distance += (norm_noise_low_resol_y - norm_y)*(norm_noise_low_resol_y - norm_y);
            distance = sqrt(distance);
            total_error += distance;

			
			//Projecting the centroid pixel from low resolution to high resolution (From January 2021):
			//A) with respect to the actual pixel location on the low resolution image and resolution reduction factor
			//low_resol_actual_centroid_x = noise_centroid_x + local_coordinates[1] + 1;
			//low_resol_actual_centroid_y = noise_centroid_y + local_coordinates[0] + 1;
			//projected_x = (int)(low_resol_actual_centroid_x*resolution_reduction_factor);   //Code updated on March 03rd, 2021
			//projected_y = (int)(low_resol_actual_centroid_y*resolution_reduction_factor);   //Code updated on March 03rd, 2021

			//B) based on the normalized coordinate on low reoslution image and padded high resolution image dimenison
			projected_x = norm_noise_low_resol_x*pd_img_features.imgwd;
			projected_y = norm_noise_low_resol_y*pd_img_features.imght;
			
			//Addition of value to the corresponding location
			low2high_summed_pd_sim_img.at<float>(projected_y, projected_x) += 1.0;

			//std::cout<<"Count "<<sampler_count<<"::\t Actual (row, col): ("<<global_row<<", "<<global_col<<") \t" \
			//         <<"Projected (row, col): ("<<projected_y<<", "<<projected_x<<")"<<std::endl;


            /* ====== COG calculation on low resolution GB patch after noise addition using Gaussian/3-point COG algorithm ====== */
            COG_output = gaussian_3_point_centroiding_algorithm(low_resolution_patch_img, max_pixel_value, identifier);

            //Projecting the COG pixel location from gaussian/3-point centroiding algorithm from low to high resolution image
            projected_x = (local_coordinates[1] + COG_output.values[1])*resolution_reduction_factor + COG_output.values[3];
            projected_y = (local_coordinates[0] + COG_output.values[0])*resolution_reduction_factor + COG_output.values[2];

            //std::cout<<"projected row: "<<projected_y<<", projected col: "<<projected_x<<std::endl<<std::endl;

            //Addition of the value at the corresponding location on padded reconstructed image with FPN adjusted
            low2high_summed_pd_sim_FPN_adjusted.at<float>(projected_y, projected_x) += 1.0;
			

            //Releasing temporary data in order to prevent memory leakage and prevent eating away of the entire RAM during the simulation
            patch_img.release();
            pd_smooth_ROI.release();
            low_resolution_patch_img.release();
            pd_lower_smooth_ROI.release();
            low_resolution_snp_patch.release();

            //Release the dynamically allocated memory in form of pointer
            //delete []COG_output;
        }
        //std::cout<<std::endl;

        sampler_count++;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* =========== Processing of the deviations in Centroiding Method =========== */
	double temp_discrete_error 			= discretization_error;
	double temp_dxdy 					= dxdy_centerofmass_avg;
	double temp_total_error 			= total_error;
	
    int temp_criterion_success_counter 	= criterion_success_counter;
	
    //int temp_invalid_counter = COM_invalid_position_counter;
	//MPI_Reduce(&temp_discrete_error, &discretization_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//MPI_Reduce(&temp_dxdy, &dxdy_centerofmass_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&temp_total_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&temp_criterion_success_counter, &criterion_success_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&temp_invalid_counter, &COM_invalid_position_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        //dxdy_centerofmass_avg = sqrt(dxdy_centerofmass_avg/(max_sampler_count - COM_invalid_position_counter));
		//discretization_error /= max_sampler_count;
		//dxdy_centerofmass_avg /= max_sampler_count;
		total_error /= max_sampler_count;
        //std::cout<<"\tANALYSIS OUTPUT FOR SIMULATION RUN:: DE = "<<discretization_error*low_resolution_dim \
		//		 <<"; COM_deviation = "<<dxdy_centerofmass_avg*low_resolution_dim<<"; Total Error = "<<total_error*low_resolution_dim<<std::endl;
        std::cout<<"\tANALYSIS OUTPUT FOR SIMULATION RUN:: Total Error = "<<total_error*low_resolution_dim_x<<std::endl;
    }


    //saving the smooth output image
    summed_smooth_img           = summed_pd_smooth_img(cv::Rect(pad_length_x, pad_length_y,
                                                            in_img_features.imgwd, in_img_features.imght));
	low2high_summed_sim_img		= low2high_summed_pd_sim_img(cv::Rect(pad_length_x, pad_length_y,
																		in_img_features.imgwd, in_img_features.imght));
	low2high_summed_sim_FPN_adjusted = low2high_summed_pd_sim_FPN_adjusted(cv::Rect(pad_length_x, pad_length_y,
                                                                                    in_img_features.imgwd, in_img_features.imght));
    //low2high_summed_sim_img     = low2high_summed_pd_sim_img(cv::Rect(pad_length_x, pad_length_y,
    //                                                          in_img_features.imgwd, in_img_features.imght));         //this is so because the Rvalue image is not padded
    low_resolution_smooth_img   = low_resolution_pd_smooth_img(cv::Rect(pad_length_low_resolution_x, pad_length_low_resolution_y,
                                                                        low_resolution_dim_x, low_resolution_dim_y));

    cv::Mat *images = new cv::Mat[num_output_frames];
    images[0] = summed_sim_img;
    images[1] = summed_smooth_img;
    images[2] = low2high_summed_sim_img;
    images[3] = low2high_summed_sim_FPN_adjusted;
    images[4] = low_resolution_smooth_img;

    summed_sim_img.release();
    summed_smooth_img.release();
    low2high_summed_sim_img.release();
    low2high_summed_sim_FPN_adjusted.release();
    low_resolution_smooth_img.release();
    summed_pd_smooth_img.release();
    low2high_summed_pd_sim_img.release();
    low2high_summed_pd_sim_FPN_adjusted.release();
    low_resolution_pd_smooth_img.release();

    std::cout<<"\t\tFrom "<<__FILE__<<"("<<__LINE__<<"):: rank "<<rank<<": Ending the simulation task"<<std::endl;

    return images;
}

//Explicit instantiation of the template methods to avoid compilation error
template cv::Mat* generate_frame<int, uchar>(   cv::Mat input_img, struct img_params in_img_features,
                                                int rank, int size, int max_pixel_value, uchar identifier);
template cv::Mat* generate_frame<int, ushort>(  cv::Mat input_img, struct img_params in_img_features,
                                                int rank, int size, int max_pixel_value, ushort identifier);
template cv::Mat* generate_frame<float, float>( cv::Mat input_img, struct img_params in_img_features,
                                                int rank, int size, float max_pixel_value, float identifier);
