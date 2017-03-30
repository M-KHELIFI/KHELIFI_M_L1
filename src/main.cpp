#include <math.h>
#include "jpg.h"
#include "mnist.h"
#include <limits.h>

float distance_sq( float* V1, float* V2){
	float d = 0;
	for(int i=0; i<784; i++) {
		d += (V1[i]-V2[i])*(V1[i]-V2[i]);
	}
	return d;
 }
float linear_classifier(float*W, float* X){
	float d= 0;
	for (int i=0; i<784; i++){
			d += W[i]*X[i];
		}
	if (d>=0) return 1;
	else return 0;
}
 
int main()
{
	float Error = 0;
    float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	float* 	test_labels = read_labels("t10k-labels.idx1-ubyte");
	float* w=new float [784];
	for (int i = 0; i<784; i++) w[i]=(float)rand()*2/INT_MAX-1;
	
	for(int i=0; i<10000; i++) {
			printf("%u\n",i);
			
				
		int inference = linear_classifier (w, test_images[i]);  
		//if(inference != (int)test_labels[i]) Error++;
		//printf("le pourcentage d'erreur est %f %u == %u %0.2f %% \n",Error, inference, (int)test_labels[i], (Error*100)/(i+1));
			
		
	}

	
	
	
    return 0;
}


