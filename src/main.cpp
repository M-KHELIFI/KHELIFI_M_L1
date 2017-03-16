#include <math.h>
#include "jpg.h"
#include "mnist.h"


float distance_sq( float* V1, float* V2){
	float d = 0;
	for(int i=0; i<784; i++) {
		d += (V1[i]-V2[i])*(V2[i]-V1[i]);
	}
	return d;
 }


int main()
{

    float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");

	for(int i=0; i<10000; i++) {
		printf("%u\n",i);
		float mind = -1;
		int NN;
		for(int j=0; j<60000; j++) {
			float d = distance_sq(test_images[i], images[j]);
			if(d<mind || mind==-1){
				mind = d;
				NN =j;
			}
		}			
		int inference = labels[NN];
		save_jpg(test_images[i], 28, 28, "%u/%u.jpg", inference, i);
	}

	
	
	
    return 0;
}


