void checkCUDAError(const char *msg);

void print_time_message(clock_t* t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	printf("%s: %f sec.\n", message, diff);
	*t1 = clock();
}
double getElapsetTime(clock_t* t1) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	*t1 = clock();
	return diff;
}
double getTotalElapsetTime(clock_t* t1) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	return diff;
}
void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

void saveSolutionIntoFile(const char *msg, float* x, int n,
		int* nodeDescription, float treshHold) {
	int writtenBars = 0;
	FILE *fp;
	fp = fopen(msg, "w");
	for (int i = 0; i < n; i++) {
		if (abs(x[i]) > treshHold) {
			writtenBars++;
			fprintf(fp, "%d,%d,%d,%d,%f\n", nodeDescription[i * 4],
					nodeDescription[i * 4 + 1], nodeDescription[i * 4 + 2],
					nodeDescription[i * 4 + 3], x[i]);
		}
	}
	fclose(fp);
	printf("Number of written bars:%d\n", writtenBars);
}

double computeTTDObjectiveValue(float* A_TTD, int* Row_IDX, int* Col_IDX,
		float* x, int n, int m, float* b,int nnz,float lambda) {
	double objectiveValue = 0;
	double residuals[m];
	for (int i = 0; i < m; i++) {
		residuals[i] = -b[i];
	}
	for (int i=0;i<nnz;i++)
	{
		residuals[Row_IDX[i]-1]+=A_TTD[i]*x[Col_IDX[i]-1];
	}
	double g_sq = 0;
	for (int i=0;i<m;i++)
	{
		g_sq+=residuals[i]*residuals[i];
	}
	double x_norm = 0;
	for (int i=0;i<n;i++)
	{
		x_norm+=abs(x[i]);
	}
	objectiveValue=x_norm*lambda+0.5*g_sq;


	return objectiveValue;
}


void updateG(float* A_TTD, int* Row_IDX, int* Col_IDX,
		float* x, int n, int m, float* b,int nnz,  float* g) {
	double residuals[m];
	for (int i = 0; i < m; i++) {
		residuals[i] = -b[i];
	}
	for (int i=0;i<nnz;i++)
	{
		residuals[Row_IDX[i]-1]+=A_TTD[i]*x[Col_IDX[i]-1];
	}
	for (int i=0;i<m;i++)
	{
		g[i]=residuals[i];
	}


}
