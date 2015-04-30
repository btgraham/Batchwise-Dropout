// https://gist.github.com/ashwin/2652488#file-cudaerrorcheck-cu
// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError() { __cudaCheckError( __FILE__, __LINE__ ); }

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
#endif
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
#endif

  return;
}

cublasHandle_t cublasHandle;
#define NTHREADS 512
#define KERNELBLOCKSIZE 32

static void cublasError(cublasStatus_t error,const char* file = 0, int linenumber = 0)
{
  switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
      break;

    case CUBLAS_STATUS_NOT_INITIALIZED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_NOT_INITIALIZED\n";
      break;

    case CUBLAS_STATUS_ALLOC_FAILED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_ALLOC_FAILED\n";
      break;

    case CUBLAS_STATUS_INVALID_VALUE:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_INVALID_VALUE\n";
      break;

    case CUBLAS_STATUS_ARCH_MISMATCH:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_ARCH_MISMATCH\n";
      break;

    case CUBLAS_STATUS_MAPPING_ERROR:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_MAPPING_ERROR\n";
      break;

    case CUBLAS_STATUS_EXECUTION_FAILED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_EXECUTION_FAILED\n";
      break;

    case CUBLAS_STATUS_INTERNAL_ERROR:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_INTERNAL_ERROR\n";
      break;
    }
}

void initializeGPU(int cudaDevice) { //pciBusID, or -1 for the first device
  int nGPU;
  bool setGPU=false;
  cudaSafeCall(cudaGetDeviceCount(&nGPU));
  for (int i=0;i<nGPU;i++) {
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, i));
    if (i==0 and cudaDevice==-1)
      cudaDevice=prop.pciBusID;

    if (prop.pciBusID==cudaDevice) {
      cout << "*";
      cudaSafeCall(cudaSetDevice(i));
      setGPU=true;
    } else {
      cout << " ";
    }
    cout << prop.pciBusID << " " << prop.name<< endl;
  }
  assert(setGPU);
  cublasError(cublasCreate(&cublasHandle),__FILE__,__LINE__);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. /////////////////////////////////////////////////////////// //////////////////////////////////////////////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
                                    float* A, float* B, float* C,
                                    int l, int m, int r,
                                    float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N,r,l,m,&alpha,B,r,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T,r,l,m,&alpha,B,r,A,l,&beta,C,r), file, linenumber);
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N,r,l,m,&alpha,B,m,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
                                      float* A, float* B, float* C,
                                      int l, int m, int r,
                                      float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T,r,l,m,&alpha,B,m,A,l,&beta,C,r), file, linenumber);
}
///////////////////////////////////////////////////////////////////////////////////////////////////


//                 _              _____ _    _ _____
//                | |            / ____| |  | |  __ \   /\
// __   _____  ___| |_ ___  _ __| |    | |  | | |  | | /  \
// \ \ / / _ \/ __| __/ _ \| '__| |    | |  | | |  | |/ /\ \
//  \ V /  __/ (__| || (_) | |  | |____| |__| | |__| / ____ \
//   \_/ \___|\___|\__\___/|_|   \_____|\____/|_____/_/    \_\
//
//
//"Unify" CPU and GPU memory

template <typename t> class vectorCUDA {
private:
  t* d_vec;
  int dsize; //When on GPU
  std::vector<t> vec;
public:
  bool onGPU;
  void copyToCPU() {
    if (onGPU) {
      onGPU=false;
      if (dsize>0)  {
        vec.resize(dsize);
        cudaSafeCall(cudaMemcpy(&vec[0],d_vec,sizeof(t)*dsize,cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaFree(d_vec));
      }
    }
  }
  void copyToGPU() {
    if (!onGPU) {
      onGPU=true;
      if (vec.size()>0)  {
        dsize=vec.size();
        cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
        cudaSafeCall(cudaMemcpy(d_vec,&vec[0],sizeof(t)*dsize,cudaMemcpyHostToDevice));
        vec.clear();
      } else {
        dsize=0;
      }
    }
  }
  void copyToGPU(cudaStream_t stream) {
    if (!onGPU) {
      onGPU=true;
      if (vec.size()>0)  {
        dsize=vec.size();
        cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
        cudaSafeCall(cudaMemcpyAsync(d_vec,&vec[0],sizeof(t)*dsize,cudaMemcpyHostToDevice,stream));
        vec.clear();
      }
    }
  }
  t*& dPtr() {
    copyToGPU();
    return d_vec;
  }
  vector<t>& hVector() {
    copyToCPU();
    return vec;
  }
  int size() {
    if (onGPU) return dsize;
    return vec.size();
  }
  float meanAbs() {
    float total=0;
    for (int i=0;i<size();i++)
      total+=fabs(hVector()[i]);
    if (total!=total) exit(1);
    return total/size();
  }
  void setZero() {
    if (onGPU) {
      cudaSafeCall(cudaMemset(d_vec,  0,sizeof(t)*dsize));
    } else {
      memset(&vec[0],0,sizeof(t)*vec.size());
    }
  }
  void setConstant(float a=0) {
    copyToCPU();
    for (int i=0;i<vec.size();i++)
      vec[i]=a;
  }
  void setUniform(float a=-0.1,float b=0.1) {
    RNG rng;
    copyToCPU();
    for (int i=0;i<vec.size();i++)
      vec[i]=rng.uniform(a,b);
  }
  void setBernoulli(float p) {
    RNG rng;
    copyToCPU();
    for (int i=0;i<vec.size();i++)
      vec[i]=rng.bernoulli(p);
  }
  void setNormal(float mean=0, float sd=1) {
    RNG rng;
    copyToCPU();
    for (int i=0;i<vec.size();i++)
      vec[i]=rng.normal(mean,sd);
  }
  void resize(int n) {
    if (onGPU) {
      if (dsize!=n) {
        if (dsize>0)
          cudaSafeCall(cudaFree(d_vec));
        if (n>0)
          cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*n));
        dsize=n;
      }
    } else {
      vec.resize(n);
    }
  }
  vectorCUDA(bool onGPU=true, int dsize=0) : onGPU(onGPU), dsize(dsize) {
    if (onGPU && dsize>0) {
      cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
    } else {
      vec.resize(dsize);
    }
  }
  ~vectorCUDA() {
    if (onGPU && dsize>0)
      cudaSafeCall(cudaFree(d_vec));
  }
  void printSubset(const char *name, int nCol,int maxPrint=10) {
    RNG rng;
    copyToCPU();
    int nRow=vec.size()/nCol;
    cout << name << " " << nRow << " " << nCol << endl;
    vector<int> rr=rng.NchooseM(nRow,min(maxPrint,nRow));
    vector<int> rc=rng.NchooseM(nCol,min(maxPrint,nCol));
    for (int i=0;i<rr.size(); i++) {
      for (int j=0;j<rc.size(); j++) {
        cout.precision(3);
        cout <<scientific<< vec[rr[i]*nCol+rc[j]] << "\t";
      }
      cout << endl;
    }
    cout << "---------------------------------------"<<endl;
  }
};


vector<int> range(int n) {
  vector<int> ret(n);
  for (int i=0; i<n; i++)
    ret[i]=i;
  return ret;
}

#ifndef NAG_MU
#define NAG_MU 0.9
#endif
__global__ void dGradientDescentNAG
(float* d_delta, float* d_momentum, float* d_weights, int nOut, float learningRate) {
  int i=blockIdx.x*nOut;
  for(int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    float w=d_weights[j];
    float m=d_momentum[j];
    float delta=learningRate*(1-NAG_MU)*d_delta[j];
    w-=m*NAG_MU;
    m=NAG_MU*m-delta;
    w+=m*(1+NAG_MU);
    d_weights[j]=w;
    d_momentum[j]=m;
  }
}

__global__ void dBound
(float* weights, float* biases, int nIn, int nOut, float bound) {
  int m=nIn*nOut;
  float acc=powf(biases[blockIdx.x],2);
  for(int i=blockIdx.x; i<m; i+=nOut)
    acc+=powf(weights[i],2);
  acc=powf(acc,0.5)/bound;
  if (acc>1) {
    biases[blockIdx.x]/=acc;
    for(int i=blockIdx.x; i<m; i+=nOut)
      weights[i]/=acc;
  }
}

__global__ void dShrinkMatrixForDropout
(float* m, float* md,
 int* inFeaturesPresent, int* outFeaturesPresent,
 int nOut, int nOutDropout) {
  int i=blockIdx.x*nOutDropout;
  int ii=inFeaturesPresent[blockIdx.x]*nOut;
  for(int j=threadIdx.x; j<nOutDropout; j+=KERNELBLOCKSIZE) {
    int jj=outFeaturesPresent[j];
    md[i+j]=m[ii+jj];
  }
}

__global__ void dShrinkVectorForDropout(float* m, float* md, int* outFeaturesPresent, int nOut, int nOutDropout) {
  for(int i=threadIdx.x; i<nOutDropout; i+=NTHREADS) {
    md[i]=m[outFeaturesPresent[i]];
  }
}

__global__ void dGradientDescentMatrixNAGlite
(float* d_delta, float* d_momentum, float* d_weights,
 int nOut, int nOutDropout,
 int* inFeaturesPresent, int* outFeaturesPresent,
 float learningRate) {
  int i=blockIdx.x*nOutDropout;
  int ii=inFeaturesPresent[blockIdx.x]*nOut;
  for(int j=threadIdx.x; j<nOutDropout; j+=KERNELBLOCKSIZE) {
    int iijj=ii+outFeaturesPresent[j];
    //NAG light
    float m=d_momentum[iijj];
    float w=d_weights[iijj];
    float delta=learningRate*(1-NAG_MU)*d_delta[i+j];
    w-=m*NAG_MU;
    m=NAG_MU*m-delta;
    w+=m*(1+NAG_MU);
    d_momentum[iijj]=m;
    d_weights[iijj]=w;
  }
}

__global__ void dGradientDescentVectorNAGlite
(float* d_delta, float* d_momentum, float* d_weights,
 int nOut, int nOutDropout,
 int* outFeaturesPresent,
 float learningRate) {
  for(int i=threadIdx.x; i<nOutDropout; i+=NTHREADS) {
    int ii=outFeaturesPresent[i];
    //NAG light
    d_weights[ii]-=d_momentum[ii]*NAG_MU;
    d_momentum[ii]=NAG_MU*d_momentum[ii]-learningRate*(1-NAG_MU)*d_delta[i];
    d_weights[ii]+=d_momentum[ii]*(1+NAG_MU);
  }
}


__global__ void dColumnSum
(float* matrix, float* target, int nRows, int nColumns) {
  int i=blockIdx.x*KERNELBLOCKSIZE+threadIdx.x;
  float t=0;
  for (int j=blockIdx.y;j<nRows;j+=KERNELBLOCKSIZE)
    t+=matrix[j*nColumns+i];
  atomicAdd(&target[i],t);
}
void columnSum(float* matrix, float* target, int nRows, int nColumns) {
  if (nColumns/KERNELBLOCKSIZE>0)
    dColumnSum<<<dim3(nColumns/KERNELBLOCKSIZE,KERNELBLOCKSIZE),KERNELBLOCKSIZE>>>(matrix, target, nRows, nColumns);
  if (nColumns%KERNELBLOCKSIZE>0) {
    int o=nColumns/KERNELBLOCKSIZE*KERNELBLOCKSIZE;
    dColumnSum<<<dim3(1,KERNELBLOCKSIZE),nColumns-o>>>(matrix+o, target+o, nRows, nColumns);
  }
  cudaCheckError();
}

__global__ void dReplicateArray
(float* src, float* dst, int nColumns) {
  int i=blockIdx.x*nColumns;
  for (int j=threadIdx.x;j<nColumns;j+=KERNELBLOCKSIZE)
    dst[i+j]=src[j];
}
void replicateArray(float* src, float* dst, int nRows, int nColumns) {
  int processed=0;
  while (processed<nRows) {
    int batch=min(32768,nRows-processed);
    dReplicateArray<<<batch,KERNELBLOCKSIZE>>>
      (src, dst+processed*nColumns, nColumns);
    processed+=batch;
  }
  cudaCheckError();
}
