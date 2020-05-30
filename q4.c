#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<math.h>

void main()
{
   
  int n=10000,i;
  double x;
  char fname[100];
  FILE *fptr;
  sprintf(fname,"q4_plot_data.dat");
  fptr=fopen(fname,"w");
  for (i = 0; i < n; i += 1)
  {
    x=(double)rand() / (double)((unsigned)RAND_MAX + 1);
    fprintf(fptr,"%lf %lf\n",x,(-1/2.0)*(log(x)));
  }
}
