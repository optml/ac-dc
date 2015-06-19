/******************************************************************************
* FILE: omp_mm.c
* DESCRIPTION:
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. Threads share row iterations
*   according to a predefined chunk size.
* AUTHOR: Blaise Barney
* LAST REVISED: 06/28/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NRA 6200                 /* number of rows in matrix A */
#define NCA 1500                 /* number of columns in matrix A */
#define NCB 300                  /* number of columns in matrix B */

void print_time_message(clock_t t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("%s: %f\n", message, diff);
}


int main (int argc, char *argv[])
{
int	tid, nthreads, i, j, k, chunk;
int threds[25];
for (i=0;i<24;i++)
  {
	  threds[i]=0;
  }
double	a[NRA][NCA],           /* matrix A to be multiplied */
	b[NCA][NCB],           /* matrix B to be multiplied */
	c[NRA][NCB];           /* result matrix C */

chunk = 10;                    /* set loop iteration chunk size */
clock_t t1, t2;
		t1 = clock();
/*** Spawn a parallel region explicitly scoping all variables ***/
#pragma omp parallel shared(a,b,c,nthreads,chunk,threds) private(tid,i,j,k)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    }
  /*** Initialize matrices ***/
  chunk=NRA/ omp_get_num_threads();
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++)
      a[i][j]= i+j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= i*j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCB; j++)
      c[i][j]= 0;

  /*** Do matrix multiply sharing iterations on outer loop ***/
  /*** Display who does which iterations for demonstration purposes ***/

  printf("Thread %d starting matrix multiply...\n",tid);




  #pragma omp for
  for (i=0; i<NRA; i++)
    {
	  threds[omp_get_thread_num()]++;
    for(j=0; j<NCB; j++)
      for (k=0; k<NCA; k++)
        c[i][j] += a[i][k] * b[k][j];
    }
  }   /*** End of parallel region ***/
  for (i=0;i<8;i++)
  {
	  printf("Thred %d workd %d \n",i,threds[i]);
  }

  print_time_message(t1, "DONE");
/*** Print results ***/


}
