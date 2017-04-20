#include <math.h>
#include "jpg.h"
#include "mnist.h"
#include <limits.h>

/*float distance_sq( float* V1, float* V2){
	float d = 0;
	for(int i=0; i<784; i++) {
		d += (V1[i]-V2[i])*(V1[i]-V2[i]);
	}
	return d;
 }*/
float linear_classifier(float*W, float* X){
	float d= 0;
	for (int i=0; i<784; i++){
			d += W[i]*X[i];
		}
	if (d>=0) return 1;
	else return -1;
}
 
int main()
{

    float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	float* 	test_labels = read_labels("t10k-labels.idx1-ubyte");
	float* w=new float [784];
	//STEP 1 : INITIALISATION
	for (int i = 0; i<784; i++) w[i]=(float)rand()*2/INT_MAX-1;

	float gamma = 0.01;
	//STEP 2 : LEARNING (que les données de train)
	for(int i=0; i<60000; i++) {
		//Calcul gradiant (g=y*x si erreur, 0 sinon)
		int prediction = linear_classifier (w, images[i]);
		int verite = (labels[i] == 1) ? 1 : -1;
			if (verite !=prediction){
				printf("Erreur\n");
				//w(t+1) = w(t) - gamma*y*x 
				for(int j=0; j<784; j++)
				w[j] = w[j] + gamma*verite*images[i][j];
			}
		}
	
	//STEP 3: TEST (que les données de test)
	float Error = 0;
	for(int i=0; i<10000; i++) {
		printf("%u\n",i);
		int inference = linear_classifier (w, test_images[i]); 
		save_jpg (test_images[i], 28, 28, "%u/%u.jpg", inference,i); 
		if((inference==1 && test_labels[i]!=1)
		||(inference ==-1 && (int)test_labels[i]==1)) Error++;
		printf("le pourcentage d'erreur est %f %u == %u %0.2f %% \n",Error, inference, (int)test_labels[i], (Error*100)/(i+1));
			
		
	}

	
	
	
    return 0;
}


