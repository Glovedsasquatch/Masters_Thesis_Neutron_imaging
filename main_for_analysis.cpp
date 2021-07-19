/* Created by raviprabhashankar on 12.10.20.
 * FILE CONTENT:
 *  - Main method to begin execution
 *  - Bifurcates the serial and the parallel execution based on the number of process(es) passed
 *    while executing mpirun
 *  - Checks execution time of the codes
 *
 * PARALLELISATION:
 *  - The parallelism is done in such a way that the number of sampling is divide equally between the
 *    processes and then added to get the final output
 *
 *  Final Outcome:
 *  - Fine granularity is causing the threads to spend more time communicating than computing, hence
 *    needs a major change
 *
 * COMMANDS FOR PROGRAM EXECUTION (all .cpp files need to be included):
 * ~$ mpic++ -ggdb -o main main.cpp <all_other_cppfiles>.cpp `pkg-config --cflags --libs opencv opencv4`
 * ~$ mpirun -n <number of processes> ./<name_for_the_executable> <Image_filename>
 *
 * */

#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <bits/stdc++.h> // this and the following two commands for creating directory in the program
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>
#include "init_params.h"
#include "run_simulation.h"
#include "parallel.h"
#include "parameter_analysis.h"


int main(int argc, char* argv[]){
    //Check for input command correctness and assign value to the image category
    check_execution(argc, argv);

    //Initialize MPI
    MPI_Init(&argc, &argv);

    //Assigning the values of rank and size
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Setting parameter files and the initializing parameters based on the file
    std::string parameters_init = std::string(argv[2]);
    readParameters(parameters_init, rank);

    //Process resource instantiation and initialization
    cv::Mat* images;
    cv::Mat input_img;
    double timestart, timeend, timetaken;                      //-------------------- Execution time start and end in seconds

    //Open input image by all the processes
    input_img = openImage(std::string(argv[1]));

    //Displaying input image for cross-checking
    if(rank == 0){
        displayImage(input_img, "INPUT IMAGE WINDOW");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //Instantiate image attributes and assigning maximum pixel value
    struct img_params in_img_features = img_data(input_img);

    //Setting image range or rescaling the input image
    if(img_category == 0){
        input_img = setImgRange(input_img, max_pixel_val._i_value, rank, uchar_identifier);
    }else if(img_category == 2) {
        input_img = setImgRange(input_img, max_pixel_val._i_value, rank, ushort_identifier);
    }else if(img_category == 5){
        input_img = setImgRange(input_img, max_pixel_val._f_value, rank, float_identifier);
    }

    //Setting/Checking the gaussian blur, sampling and noise parameters as per constraints
    setGBparameters();
    setSamplingParameters(input_img, size, rank);
    setNoiseParameters();


    if(rank == 0){
        //Output to the terminal to check the image type
        std::cout<<"From "<<__FILE__<<"("<<__LINE__<<"):: The input image size is as follows:\n" \
                 << "\t Image resolution: \t" << in_img_features.imght << "x" << in_img_features.imgwd \
                 << "\n\t Image type: \t\t" << in_img_features.imgtyp\
                 << "\n\t Number of channel(s): \t" << in_img_features.imgchannel \
                 << "\n\t Image bit-size: \t" << in_img_features.imgbitsize << std::endl;
        std::cout<<std::string(star_separator_count, '*')<<std::endl<<std::endl;


        //Output to a file for parameter check/debug
        std::ofstream parameters_check_file("parameters_check_file.txt", std::ios::out);
        parameters_check_file <<"From the file: "<<__FILE__<<": FOR DEBUGGING --\n\n Value of Gaussian parameters::\n" \
                          <<std::setw(50)<<"Kernel size = "<<gb_kernel_x<<" "<<gb_kernel_y<<std::endl \
                          <<std::setw(50)<<"Sigma value = "<<gb_sigma<<std::endl<<std::endl \

                          <<"Input image pixel value related parameters::\n" \
                          <<std::setw(50)<<"Image type/category = "<<in_img_features.imgbitsize<<"-bit"<<std::endl;
        if(img_category == 0 || img_category ==2){
            parameters_check_file <<std::setw(50)<<"max. pixel value = "<<max_pixel_val._i_value<<std::endl;
            parameters_check_file <<std::setw(50)<<"Scaled lower and upper bound pixel values = "<<max_pixel_val._i_value*lower_range_percent \
                              <<" and "<<max_pixel_val._i_value*upper_range_percent<<std::endl<<std::endl;
        } else{
            parameters_check_file <<std::setw(50)<<"max. pixel value = "<<max_pixel_val._f_value<<std::endl;
            parameters_check_file <<std::setw(50)<<"Scaled lower and upper bound pixel values = "<<max_pixel_val._f_value*lower_range_percent \
                              <<" and "<<max_pixel_val._f_value*upper_range_percent<<std::endl<<std::endl;
        }
        parameters_check_file <<"Sampling related parameters::\n" \
                              <<std::setw(50)<<"max sampling per image = " <<max_sampler_count<<std::endl \
                              <<std::setw(50)<<"resolution reduction factor = "<<resolution_reduction_factor<<std::endl \
                              <<std::setw(50)<<"noise present = "<<noise_present<<std::endl;
        if(noise_present == 1){
            parameters_check_file <<std::setw(50)<<"min. and max. percent noise = "<<percent_min_noise<<" and "<<percent_max_noise<<std::endl;
            if(img_category == 0 || img_category ==2){
                parameters_check_file <<std::setw(50)<<"Lower and Upper bound noise values = "<<lower_noise_limit._i_noise \
                                      <<" and "<<upper_noise_limit._i_noise<<std::endl<<std::endl;
            } else{
                parameters_check_file <<std::setw(50)<<"Lower and Upper bound noise values = "<<lower_noise_limit._f_noise \
                                      <<" and "<<upper_noise_limit._f_noise<<std::endl<<std::endl;
            }
        }

        parameters_check_file <<"Output Related::\n" \
                              <<std::setw(50)<<"Total Output Image frames = "<<num_output_frames<<std::endl<<std::endl;
        parameters_check_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);







    //Record the start clock tick
    timestart = MPI_Wtime();

    //Run simulation to generate frames
    if(rank == 0){
        std::cout<<"\tRunning simulation to generate the image frames....."<<std::endl;
    }

    if(img_category == 0){
        images = generate_frame(input_img, in_img_features, rank, size, max_pixel_val._i_value, uchar_identifier);
    }else if(img_category == 2){
        images = generate_frame(input_img, in_img_features, rank, size, max_pixel_val._i_value, ushort_identifier);
    }else if(img_category == 5){
        images = generate_frame(input_img, in_img_features, rank, size, max_pixel_val._f_value, float_identifier);
    }

    //Checking empty frames
    check_empty_simulationframes(images, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);


    //Record the end clock tick
    timeend = MPI_Wtime();
    timetaken = double(timeend - timestart);


    //Prompt to mark the end of the simulation for image frame generation
    if(rank == 0){
        std::cout<<"\tSimulation finished:: Total execution time: "<<std::fixed<<std::setprecision(3)<<timetaken<<" seconds"<<std::endl;
        std::cout<<std::string(star_separator_count, '*')<<std::endl<<std::endl;
    }


    /* ============================== Processing storing the image output frames ================================== */
	//creating folder using rank 0
	std::string dir_name = "./Comparison_results/current_run/Image_";
	if(rank == 0){		
		std::string temp;
		for(int index = 0; index < num_output_frames; index++){
			temp = dir_name + std::to_string(index+1);
			if(mkdir(temp.c_str(), 0777) == -1){
				std::cerr<<"From rank "<<rank<<":: Status/Error on folder name "<<temp<<": "<<std::strerror(errno)<<std::endl;
			} else {
				std::cout<<"From rank "<<rank<<":: Directory "<<temp<<" created!"<<std::endl;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Filename prefix instantiation and assignment
    std::string simulated_output = dir_name + "1/simulated_output"
								   + "_ns" + std::to_string(max_sampler_count)
                                   + "_ks" + std::to_string(gb_kernel_x)
								   + "_nup" + std::to_string((int)percent_max_noise*100)
                                   + "_" + std::to_string(in_img_features.imgwd);
    std::string smoothed_output = dir_name + "2/smoothed_output"
                                  + "_ns"+ std::to_string(max_sampler_count)
                                  + "_ks" + std::to_string(gb_kernel_x)
								  + "_nup" + std::to_string((int)(percent_max_noise*100))
                                  + "_" + std::to_string(in_img_features.imgwd);
    std::string simulated_with_noise_output = dir_name + "3/simulated_with_noise_output"
											  + "_ns" + std::to_string(max_sampler_count)
                                              + "_ks" + std::to_string(gb_kernel_x)
											  + "_nup" + std::to_string((int)(percent_max_noise*100))
                                              + "_" + std::to_string(in_img_features.imgwd);
    std::string low_resolution_smoothed_output = dir_name + "4/low_resolution_smoothed_output"
												 + "_ns" + std::to_string(max_sampler_count)
                                                 + "_ks" + std::to_string(gb_kernel_x)
												 + "_nup" + std::to_string((int)(percent_max_noise*100))
                                                 + "_" + std::to_string(in_img_features.imgwd / resolution_reduction_factor);


    //Adding filename suffix based on the image category
    if (img_category == 0) {
        simulated_output = simulated_output + "_" + std::to_string(8) + "-bit_rank_" + std::to_string(rank) + ".tif";
        smoothed_output = smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        simulated_with_noise_output = simulated_with_noise_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        low_resolution_smoothed_output = low_resolution_smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
    } else if (img_category == 2) {
        simulated_output = simulated_output + "_" + std::to_string(16) + "-bit_rank_" + std::to_string(rank) + ".tif";
        smoothed_output = smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        simulated_with_noise_output = simulated_with_noise_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        low_resolution_smoothed_output = low_resolution_smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
    } else if (img_category == 5) {
        simulated_output = simulated_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        smoothed_output = smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        simulated_with_noise_output = simulated_with_noise_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
        low_resolution_smoothed_output = low_resolution_smoothed_output + "_" + std::to_string(32) + "-bit_rank_" + std::to_string(rank) + ".tif";
    }


    //saving the simulated and smoothed output image
    cv::imwrite(simulated_output, images[0]);
    cv::imwrite(smoothed_output, images[1]);
    cv::imwrite(simulated_with_noise_output, images[2]);
    cv::imwrite(low_resolution_smoothed_output, images[3]);


    //releasing the resources
    images[0].release();
    images[1].release();
    images[2].release();
    images[3].release();


    //clearing cin buffer
    std::cin.clear();
    std::cin.sync();

	char choice{};
	if(rank == 0){
		std::cout<<"Want to run the analysis? (y/n): ";
		std::cin>>choice;
	}
	MPI_Bcast(&choice, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	if(choice != 'y'){
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}else{
		//Preparing Analysis Environment
	    setAnalysisEnvironment(in_img_features);

	    if(rank == 0){
	        //Call to initialize the file with the parameter values and the headers of the dependent and independent parameters
	        init_analysisOutputFile(independent_param, dependent_param);

	        //Actual Execution of the parameter analysis for the current run
	        std::cout<<__FILE__<<"("<<__LINE__<<"):: Begin the analysis for the parameter: "<<dependent_param.param_name \
	             <<" v/s "<<independent_param.param_name<<std::endl;
	    }
	    //Synchronizing the processes
	    MPI_Barrier(MPI_COMM_WORLD);

	    timestart = MPI_Wtime(); //record time for the start of the analysis
	    if(img_category == 0){
	        executeAnalysis(input_img, in_img_features, rank, size, uchar_identifier);
	    }else if(img_category == 2){
	        executeAnalysis(input_img, in_img_features, rank, size, ushort_identifier);
	    }else if(img_category == 5){
	        executeAnalysis(input_img, in_img_features, rank, size, float_identifier);
	    }
	    timeend = MPI_Wtime(); //record time for the end of the analysis

	    //Calculation of total time spend in analysis and printing and Safe copy of final analysis output file
	    if(rank == 0){
	        //Execution time in analysis
	        timetaken = double(timeend - timestart);
	        std::cout<<"\tAnalysis finished:: Total execution time: "<<std::fixed<<std::setprecision(3)<<timetaken <<" seconds"<<std::endl;

	        //Safecopy of final analysis output file from temp to actual file
	        safecopy_final_output_analysis();
	    }	
	}
    

    //releasing the resources
    input_img.release();

    //Finalizing the processes
    MPI_Finalize();

    return 0;
}