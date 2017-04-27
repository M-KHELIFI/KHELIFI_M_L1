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
	else return -1;
}
const int K= 10;
float A[K][784];
float B[K][784];

int main()
{

    float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	float* 	test_labels = read_labels("t10k-labels.idx1-ubyte");
	float* w=new float [784];
	//KMEANS

int* n=new int [K];

	for (int i=0;i<K;i++){
			n[i]=0;
			for(int j=0;j<784;j++) {
			A[i][j] = (float)rand()*2/INT_MAX-1;
			B[i][j] = 0;
			}
	}
	for (int t=0;t<1000;t++){	
		for (int i=0;i<K;i++){
			n[i]=0;
			for(int j=0;j<784;j++)B[i][j] = 0;
		}
	
	for(int i=0;i<60000;i++){
		printf("i= %u t= %u\n", i ,t);
		float mind = -1; int gagnant = 0;
		for(int k=0; k<K;k++){
			float d = distance_sq(A[k],images[i]);
			if(d<= mind|| mind==-1){
				mind = d, gagnant = k;
			}
		}
		for (int j=0; j<784; j++) B[gagnant][j] += images[i][j];
		n[gagnant]++;
		
		for(int k=0;k<K; k++) for(int j=0; j<784; j++)
			A[k][j] = B[k][j]/n[k];	

		}
		for(int k=0;k<K;k++){
			save_jpg(A[k],28,28,"%u/%04u.jpg", k, t);
		}	
	}

	//STEP 1 : INITIALISATION
	for (int i = 0; i<784; i++) w[i]=(float)rand()*2/INT_MAX-1;

	float gamma = 0.01;
	//STEP 2 : LEARNING (que les données de train)
	for(int i=0; i<100; i+=4) {
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


