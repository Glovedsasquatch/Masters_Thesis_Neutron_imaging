//
// Created by raviprabhashankar on 15.10.20.
//

#ifndef SIMULATIONIMAGES_PARALLELIZED_PARAMETER_ANALYSIS_H
#define SIMULATIONIMAGES_PARALLELIZED_PARAMETER_ANALYSIS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <mpi/mpi.h>
#include "init_params.h"
#include "important_algorithms.h"

//Structure defining the parameters
struct parameters{
    int         index;
    std::string param_name;
};
extern struct parameters dependent_param, independent_param;
//========================================================================


//File to initialize the independent parameter's values
extern std::string ind_param_filename;
//========================================================================


//Current run filename
extern std::string analysis_output_filename;
extern std::string temp_analysis_output_filename;           //intermediate file for data safety between two analysis runs to ensure
                                                            //that finalized data files previous runs are not overwritten in case the
                                                            //parameter values they were executed for was mistakenly unchanged
//========================================================================

void setAnalysisEnvironment(struct img_params);
void init_analysisOutputFile(struct parameters, struct parameters);
template<typename T>
void executeAnalysis(cv::Mat, struct img_params, int, int, T);                 //Executes the parameter analysis
template<typename D, typename T>
double dxdy_centerofmass_avg_analysis(cv::Mat, struct img_params, int, D, T);    //Analysis of deviations in COM methodology
void safecopy_final_output_analysis();                                          //Safe copy of final analysis output data from temp to actual file

#endif //SIMULATIONIMAGES_PARALLELIZED_PARAMETER_ANALYSIS_H
