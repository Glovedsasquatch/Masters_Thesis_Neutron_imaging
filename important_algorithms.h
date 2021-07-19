//
// Created by raviprabhashankar on 07.03.21.
//

#ifndef SIMULATIONIMAGES_PARALLELIZED_RUN_IMPORTANT_ALGORITHMS_H
#define SIMULATIONIMAGES_PARALLELIZED_RUN_IMPORTANT_ALGORITHMS_H

#include<iostream>
#include "init_params.h"

template<typename T>
cv::Mat coordinate_map_lowresolGB(  cv::Mat, int, int, int, int, int*, int&, int&, T);      // Coordinate Mapping Algorithm:
                                                                                            // For coordinate mapping from higher to lower resolution image;
                                                                                            // Returns the low resolution patch (neutron event) corresponding
                                                                                            // to every high resolution Gaussian event

template<typename D, typename T>
centroid_algorithm_output gaussian_3_point_centroiding_algorithm(cv::Mat, D, T);         // Gaussian/3-point Centroiding Algorithm:
                                                                    // Hybrid algorithm to maximize the DQE (Detective quantum efficiency) by ensuring a fix
                                                                    // to the Fixed Pattern Noise (FPN);
                                                                    // Returns (x_cm, y_cm): corresponding to high resolution

#endif //SIMULATIONIMAGES_CHANGE_OF_PADDING_SCHEME_IMPORTANT_ALGORITHMS_H
