# Master's Thesis: Single Particle Detection in Event-Mode Imaging using Neutrons and Photons
## Abstract
Event-Mode Imaging is a method where the final image is obtained as a summation of individually acquired particle interactions. In neutron imaging, scintillation helps in neutron detection via nuclear activation producing a secondary radiation which upon ionizing emits visible light in the shape of a cone. The low spatial resolution in fast neutron imaging results from blurring introduced by this cone shape emission of light with the spatial resolution roughly proportional to the thickness of the scintillation material. To overcome the problem of spatial locality, a center-of-mass (COM) method is used to find the most probable location of the spot for each neutron interaction, potentially allowing to increase spatial resolution and efficiency of the method.

Here we present a parametric study for event-based imaging to computationally obtain optimal parameters in a neutron imaging environment, such as the impact of noise, the size of the light spot, the deviations in the COM methodology and so on through simulations. This is done by random probabilistic sampling of pixels in a grayscale input image simulating the particle interaction and applying a Gaussian blur patch to the sampled pixel value with a kernel size to replicate the problem of low spatial resolution in the simulation. The patch image is mapped to lower resolution to obtain a simulated neutron event (SNE) which is then perturbed by a uniform additive noise. Centroiding is performed for each SNE to compare deviations and to generate a super-resolution image by mapping the centroid values back to high-resolution. The thesis also discusses the numerical bias in the centroiding causing a fixed pattern noise which is corrected using suitable numerical technique and the reconstructed images generated using the biased and unbiased algorithm are analyzed qualitatively to cross-validate the results of the quantitative analysis.


## Contents
The folder contains:
- /Comparison_results folder: program writes the output into this folder
- /Parameters_data_files folder: contains files for the input parameters
- /Sample_images folder: contains input image file (any 8/16/32-bit grayscale image with equal resolution in x and y-direction can be used)
- *.cpp and *.h files files
- parameters_check_file.txt: the file is written for the cross-checking the correctness of the input parameters

## Command
- to compile the code (with MPI parallelization): 
```~$ mpic++ -ggdb -o SimulationImages main.cpp init_params.cpp run_simulation.cpp important_algorithms.cpp parallel.cpp parameter_analysis.cpp `pkg-config --cflags --libs opencv opencv4` ```
- to execute the code: 
```~$ mpirun -n <numerber_of_processors> ./SimulationImages ./Sample_images/Base_image_1280_16.tif ./Parameters_data_files/all_initial_params.dat```
It is to be noted that the above program commands generate all the output images and hence, is used for the qualitative analysis. For the quantitative analysis, the ```main_for_analysis.cpp``` is to be changed to ```main.cpp``` followed by recompiling the program. 

## Contact
In case of any issue or queries, kindly write a mail to raviprabhashankartum@gmail.com.