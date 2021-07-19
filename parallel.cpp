/* Created by raviprabhashankar on 23.10.20.
 * RELEVANT DEPENDENCY:
 *  - parallel.h
 *
 *  IMPORTANT NOTES:
 *  - STATIC ALLOCATION of buffer can cause bad memory allocation error due to stack-based memory allocation
 *    with bad_alloc() runtime error. This is because the memory required for transferring the large image files
 *    can overflow stack. For large buffer sizes, follow DYNAMIC ALLOCATION as it is heap-based allocation
 */

#include <mpi.h>
#include "init_params.h"
#include "parallel.h"

const unsigned int MAXBYTES = 2560*2560 + 1000; 			//extra buffer given to accommodate the row, column,
                                                			//channels and image type information

int img_identifier[4] = {2, 5, 2, 5};	//0: unsigned char; 2: indicates unsigned short; 5: float 

template<typename T>
void matsend(cv::Mat m, int dest, T identifier){
	std::cout<<"\t\t\t"<<__func__<<"("<<__LINE__<<"): Sending in progress..."<<std::endl;
		
	T buffer[MAXBYTES];
	int rows = m.rows;
	int cols = m.cols;
	int type = m.type();
	int channels = m.channels();
    memcpy(&buffer[0 * sizeof(int)],(T*)&rows,sizeof(int));
	memcpy(&buffer[1 * sizeof(int)],(T*)&cols,sizeof(int));
	memcpy(&buffer[2 * sizeof(int)],(T*)&type,sizeof(int));

    //size of each pixel
    int bytespersample;
    if(type == 0){
        bytespersample = 1;
    } else if(type == 2){
        bytespersample = 2;
    } else if(type == 5){
		bytespersample = 4;
	}

    int bytes = rows*cols*channels*bytespersample;
std::cout << "matsnd: rows=" << rows << std::endl;
std::cout << "matsnd: cols=" << cols << std::endl;
std::cout << "matsnd: type=" << type << std::endl;
std::cout << "matsnd: channels=" << channels << std::endl;
std::cout << "matsnd: bytes=" << bytes << std::endl;

    if(!m.isContinuous()){
        m = m.clone();
    }

    memcpy(&buffer[3 * sizeof(int)], m.data, bytes);
    if(type == 0){
		MPI_Send(buffer, bytes+3*sizeof(int), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
	} else if(type == 2){
		MPI_Send(buffer, bytes+3*sizeof(int), MPI_UNSIGNED_SHORT, dest, 0, MPI_COMM_WORLD);
	} else if(type == 5){
		MPI_Send(buffer, bytes+3*sizeof(int), MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
	}
}

template<typename T>
cv::Mat matrecv(int src, int flag, T identifier){
    MPI_Status status;
    int count, rows, cols, channels, type;
	T buffer[MAXBYTES];

    if(flag == 0){
		MPI_Recv(buffer,sizeof(buffer),MPI_UNSIGNED_CHAR,src,0,MPI_COMM_WORLD,&status);
      	MPI_Get_count(&status,MPI_UNSIGNED_CHAR,&count);
	} else if(flag == 2){
		MPI_Recv(buffer,MAXBYTES,MPI_UNSIGNED_SHORT,src,0,MPI_COMM_WORLD,&status);
      	MPI_Get_count(&status,MPI_UNSIGNED_SHORT,&count);
	} else if(flag == 5){
		MPI_Recv(buffer,MAXBYTES,MPI_FLOAT,src,0,MPI_COMM_WORLD,&status);
      	MPI_Get_count(&status,MPI_FLOAT,&count);
	}
      
	memcpy((T*)&rows,&buffer[0 * sizeof(int)], sizeof(int));
	memcpy((T*)&cols,&buffer[1 * sizeof(int)], sizeof(int));
	memcpy((T*)&type,&buffer[2 * sizeof(int)], sizeof(int));
	
std::cout << "matrcv: Count=" << count << std::endl;
std::cout << "matrcv: rows=" << rows << std::endl;
std::cout << "matrcv: cols=" << cols << std::endl;
std::cout << "matrcv: type=" << type << std::endl;

    cv::Mat received = cv::Mat(rows, cols, type, (T*)&buffer[3 * sizeof(int)]);

    return received;
}


void add_final_frames(cv::Mat* image, int size, int rank) {
    cv::Mat received_frame;
	int current_size;
    
	for(int index = 0; index < num_output_frames; index++){
		current_size = size;
		std::cout<<"\t\t"<<__func__<<"("<<__LINE__<<"): from rank "<<rank<<":: image type "<<img_identifier[index]<<std::endl;
		
		while (current_size > 1) {
			//If the current size is even
	        if (current_size % 2 == 0) {
				if (rank < current_size / 2) { //Receiving calls for the even case
	                int src = current_size - rank - 1;
					if(img_identifier[index] == 0){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and sender rank "<<src<<" for image type "<<img_identifier[index]<<std::endl;
						received_frame = matrecv(src, img_identifier[index], uchar_identifier);
					} else if(img_identifier[index] == 2){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and sender rank "<<src<<" for image type "<<img_identifier[index]<<std::endl;
						received_frame = matrecv(src, img_identifier[index], ushort_identifier);
					} else if(img_identifier[index] == 5){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and sender rank "<<src<<" for image type "<<img_identifier[index]<<std::endl;
						received_frame = matrecv(src, img_identifier[index], float_identifier);
					}
	                
	                cv::add(image[index], received_frame, image[index], cv::noArray(), -1);
	            } else if (rank < current_size) { //Sending calls for the even case
	                int dest = current_size - rank - 1;
					if(image[index].type() == 0){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and receiving rank "<<dest<<" for image type "<<image[index].type()<<std::endl;
						matsend(image[index], dest, uchar_identifier);	
					} else if(image[index].type() == 2){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and receiving rank "<<dest<<" for image type "<<image[index].type()<<std::endl;
						matsend(image[index], dest, ushort_identifier);
					} else if(image[index].type() == 5){
						std::cout<<"\t\t\tSelf rank "<<rank<<" and receiving rank "<<dest<<" for image type "<<image[index].type()<<std::endl;
						matsend(image[index], dest, float_identifier);
					}
	                
	            }
	            current_size = current_size / 2;
	        } else {
	            if (rank == current_size - 2) {
	                int src = current_size - 1;
	                if(img_identifier[index] == 0){
						received_frame = matrecv(src, img_identifier[index], uchar_identifier);
					} else if(img_identifier[index] == 2){
						received_frame = matrecv(src, img_identifier[index], ushort_identifier);
					} else if(img_identifier[index] == 5){
						received_frame = matrecv(src, img_identifier[index], float_identifier);
					}
					
	                cv::add(image[index], received_frame, image[index], cv::noArray(), -1);
	            } else if (rank == current_size - 1) {
	                int dest = current_size - 2;
	                if(image[index].type() == 0){
						matsend(image[index], dest, uchar_identifier);	
					} else if(image[index].type() == 2){
						matsend(image[index], dest, ushort_identifier);
					} else if(image[index].type() == 5){
						matsend(image[index], dest, float_identifier);
					}
	            }
	            current_size = current_size - 1;
	        }
			MPI_Barrier(MPI_COMM_WORLD);
		}
    }
}
//Explicit Instantiation of template functions (refer: https://bytefreaks.net/programming-2/c/c-undefined-reference-to-templated-class-function)
template void matsend<uchar>(cv::Mat, int, uchar);
template cv::Mat matrecv<uchar>(int, int, uchar);

template void matsend<ushort>(cv::Mat, int, ushort);
template cv::Mat matrecv<ushort>(int, int, ushort);

template void matsend<float>(cv::Mat, int, float);
template cv::Mat matrecv<float>(int, int, float);



void check_empty_simulationframes(cv::Mat* images, int rank, int size){
	std::cout<<"\t"<<__func__<<" From rank "<<rank<<" for debugging:"<<std::endl;
    if(size == 1){
        for(int index = 0; index < num_output_frames; index++){
            if(images[index].empty()){
                std::cout<<"\n\t\t"<<__FILE__<<"["<<__FUNCTION__<<"("<<__LINE__<<")]: Output image name convention:" \
                         <<"\n\t\t\t Image 1: High Resolution Simulated Image" \
                         <<"\n\t\t\t Image 2: High Resolution Smooth Image" \
                         <<"\n\t\t\t Image 3: Low to High Resolution Simulated Image" \
                         <<"\n\t\t\t Image 4: Low Resolution Smooth Image"<<std::endl;
                std::cout<<"\n\t\t Serial Execution:: " \
                         <<": Invoking abort due to empty \'Image "<<index+1<<"\' file"<<std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    } else {
        for(int index = 0; index < num_output_frames; index++){
            if(images[index].empty()){
                std::cout<<"\n\t\t"<<__FILE__<<"["<<__FUNCTION__<<"("<<__LINE__<<")]: Output image name convention:" \
                         <<"\n\t\t\t Image 1: High Resolution Simulated Image" \
                         <<"\n\t\t\t Image 2: High Resolution Smooth Image" \
                         <<"\n\t\t\t Image 3: Low to High Resolution Simulated Image" \
                         <<"\n\t\t\t Image 4: Low Resolution Smooth Image"<<std::endl;
                std::cout<<"\n\t\t Parallel Execution:: From rank "<<rank \
                         <<": Invoking abort due to empty \'Image "<<index+1<<"\' file"<<std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
	std::cout<<"\t\tOut from "<<__func__<<" by rank "<<rank<<std::endl;
}

