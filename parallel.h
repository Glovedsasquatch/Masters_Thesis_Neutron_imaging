/* Created by raviprabhashankar on 23.10.20.
 * FILE CONTENT:
 *  - Function for MPI parallelization within OpenCV framework
 *
 * Relevant file(s):
 *  - parallel.cpp
 */

#ifndef SIMULATIONIMAGES_PARALLELIZED_PARALLEL_H
#define SIMULATIONIMAGES_PARALLELIZED_PARALLEL_H

#include <mpi.h>

extern const unsigned int MAXBYTES;                     //Buffer size for MPI_Send and MPI_Recv within OpenCV framework


template<typename T> void matsend(cv::Mat, int, T);    	//MPI_Send within OpenCV framework
template<typename T> cv::Mat matrecv(int, int, T);    	//MPI_Recv within OpenCV framework
void check_empty_simulationframes(cv::Mat*, int, int);  //Check for emptiness of returned frames
void add_final_frames(cv::Mat*, int, int);				//Addition of final image frames


#endif //SIMULATIONIMAGES_PARALLELIZED_PARALLEL_H
