/* Created by raviprabhashankar on 10.10.20.
 *FILE CONTENT:
 *  - Function for simulation of the frames based on the simulation parameters set
 *    by rank 0 of the processes (the method is executed independently by the processes)
 *
 * Relevant file(s):
 *  - run_simulation.cpp
 */

#ifndef SIMULATIONIMAGES_PARALLELIZED_RUN_SIMULATION_H
#define SIMULATIONIMAGES_PARALLELIZED_RUN_SIMULATION_H

#include <random>
#include <cmath>
#include <mpi/mpi.h>
#include "init_params.h"
#include "important_algorithms.h"

template<typename D, typename T> cv::Mat* generate_frame(cv::Mat, struct img_params, int, int, D, T); //Simulation for single frame generation
cv::Mat* generate_frame_version01(cv::Mat, std::string, int);              //Simulation for single frame generation

#endif //SIMULATIONIMAGES_PARALLELIZED_RUN_SIMULATION_H
