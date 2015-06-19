//ulimit -s unlimited
#include <stdio.h>
#include <stdlib.h>

int main(void) {

	FILE *fp;
		fp = fopen("/tmp/test.txt", "w");
	fprintf(fp, "started\n");
	printf("started\n");
	int n = 10000;
	int p = 20;
	printf("n=%d\n", n);
	fprintf(fp, "n=%d\n", n);
	printf("p=%d\n", p);
	fprintf(fp, "p=%d\n", p);

	double A[n];
	int i, j;

	fprintf(fp, "vector allocated\n");
	printf("vector allocated\n");
	for (i = 0; i < n; i++) {
			A[i] = i;
	}


	fprintf(fp, "work finished\n");
	printf("work finished\n");
	fclose(fp);
}

