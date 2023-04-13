/*The following code represents multiple approaches used to iteratively solve 
the Q and FhD components to identify the MRI voxel values through the equation :

                                (FhF + λWhW)ρ=FhD

Where the (FhF + λWhW) component is the pre-computed Q vector, ρ is the unknown vector of 
MRI voxel values, Fh is a matrix that models the physics of the imaging process, and
D is the sample data from the MRI scanner.
The goal of this parallel computing approach comparison is to generate an algorithm to 
parse through the available data (in unit millions) and calculate the Q and FhD components
in as little time as possible. 

Identifying an efficient algorithm for calculating the MRI voxels is imperative to providing 
real-time medical assessments to patients and provide treatments as soon as possible.*/

/*The following code is a collection of different kernels utilizing experimental methods
for voxel computations, data collection and pre-processing has been performed separately
The best results given were with Kernel 5 - Special Function Units (SFU) starting on line 107*/  

// Kernel 1 - Impractical Version
__global__ void cmpFhD_1(float* rPhi, iPhi, rD, iD, kx, ky, kz, x, y, z, rMu, iMu, rFhD, iFhD, int N)
{
    int m = blockIdx.x + FHD_THREADS_PER_BLOCK + threadIdx.x;
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];

    for(int n=0; n<N; n++) // N = Number of Voxels
    {
        floatexpFhD = 2*PI*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
        floatcArg = cos(expFhD); floatsArg = sin(expFhD);
        rFhD[n] += rMu[m]*cArg - iMu[m]*sArg;
        iFhD[n] += iMu[m]*cArg + rMu[m]*sArg; // large number  of conflicts expected
    }
}

// Kernel 2 - Naive Kernel - Non-Optimized
__global__ void cmpFhD_2(float* rPhi, iPhi, phiMag, kx, ky, kz, x, y, z, rMu, iMu, int M)
{
    int n = blockIdx.x + FHD_THREADS_PER_BLOCK + threadIdx.x;
    for(int m=0; m<M; m++)
    {
        float expFhD = 2*PI*(kx[m]*x[n]+ky[m]*y[n]+kz[m]*z[n]);
        float cArg = cos(expFhD);
        float sArg = sin(expFhD);
        rFhD[n] += rMu[m]*cArg - iMu[m]*sArg;
        iFhD[n] += iMu[m]*cArg + rMu[m]*sArg; // 13 operations, 14 memory accesses
    }
}

// Kernel 3 - Using Registers
__global__ void cmpFhD_3(float* rPhi, iPhi, phiMag, kx, ky, kz, x, y, z, rMu, iMu, int M)
{
    int n = blockIdx.x + FHD_THREADS_PER_BLOCK + threadIdx.x;
    float xn_r=x[n]; float yn_r=y[n]; float zn_r=z[n];
    float rFhDn_r=rFhD[n]; float iFhDn_r=iFhD[n];

    for (int m=0; m<M; m++)
    {
        float expFhD=2*PI*(kx[m]*xn_r+ky[m]*yn_r+kz[m]*zn_r);
        float cArg = cos(expFhD);
        float sArg = sin(expFhD);

        rFhDn_r+=rMu[m]*cArg - iMu[m]*sArg;
        iFhDn_r+=iMu[m]*cArg + rMu[m]*sArg;
    }
    rFhD[n]=rFhD_r; iFhD[n]=iFhD_r; // 13 operations, 7 memory accesses
}

// Kernel 4 - Using Constant Memory
// chunking k-space into constant memory with arbitrary CHUNK_SIZE
__constant__ float kx_c[CHUNK_SIZE], 
                   ky_c[CHUNK_SIZE],
                   kz_c[CHUNK_SIZE];
//...
void main(){
    for(int i=0; i<M/CHUNK_SIZE; i++);
    cudaMemcpyToSymbol(kx_c, &kx[i*CHUNK_SIZE], 4*CHUNK_SIZE, 
                        cudaMemCpyHostToDevice);
    cudaMemcpyToSymbol(kx_c, &ky[i*CHUNK_SIZE], 4*CHUNK_SIZE, 
                        cudaMemCpyHostToDevice);
    cudaMemcpyToSymbol(kx_c, &kz[i*CHUNK_SIZE], 4*CHUNK_SIZE, 
                        cudaMemCpyHostToDevice);
//...    
cmpFhD_4<<<FHD_THREADS_PER_BLOCK, N/FHD_THREADS_PER_BLOCK>>>(rPhi, iPhi, phiMag, x, y, z, rMu, 
                                                            iMu, CHUNK_SIZE);
// need to call kernel 1 more time if M is not perfect multiple of CHUNK_SIZE
}
__global__ void cmpFhD_4(float* rPhi, iPhi, phiMag, kx, ky, kz, x, y, z, rMu, iMu, int M)
{
    int n = blockIdx.x + FHD_THREADS_PER_BLOCK + threadIdx.x;
    float xn_r=x[n]; float yn_r=y[n]; float zn_r=z[n];
    float rFhDn_r=rFhD[n]; float iFhDn_r=iFhD[n];

    for (int m=0; m<M; m++)
    {
        float expFhD=2*PI*(kx_c[m]*xn_r+ky_c[m]*yn_r+kz_c[m]*zn_r);
        float cArg = cos(expFhD);
        float sArg = sin(expFhD);

        rFhDn_r+=rMu[m]*cArg - iMu[m]*sArg;
        iFhDn_r+=iMu[m]*cArg + rMu[m]*sArg;
    }
    rFhD[n]=rFhD_r; iFhD[n]=iFhD_r; 
// assumes 32 threads/wrap, eliminates 31/32 of global memory acess, compute to memory access ratio now 13:2
}

// Kernel 5 - Special Function Units - BEST RESULTS
__global__ void cmpFhD_5(float* rPhi, iPhi, phiMag, x, y, z, rMu, iMu, int M)
{
    int n = blockIdx.x + FHD_THREADS_PER_BLOCK + threadIdx.x;
    float xn_r=x[n]; float yn_r=y[n]; float zn_r=z[n];
    float rFhDn_r=rFhD[n]; float iFhDn_r=iFhD[n];

    for (int m=0; m<M; m++)
    {
        float expFhD=2*PI*(kx_c[m]*xn_r+ky_c[m]*yn_r+kz_c[m]*zn_r);
        float cArg = __cos(expFhD); // hardware intrinsic functions
        float sArg = __sin(expFhD);

        rFhDn_r+=rMu[m]*cArg - iMu[m]*sArg;
        iFhDn_r+=iMu[m]*cArg + rMu[m]*sArg;
    }
    rFhD[n]=rFhD_r; iFhD[n]=iFhD_r; 
}