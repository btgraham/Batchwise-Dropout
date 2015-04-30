#include <curand_kernel.h>
#define CURANDBLOCKS 32
__global__ void dCurandInit(curandState *d_state) {
  curand_init(blockIdx.x*KERNELBLOCKSIZE+threadIdx.x,0,0,&d_state[blockIdx.x*KERNELBLOCKSIZE+threadIdx.x]);
}
__global__ void dCurandBernoulliDrop
(curandState *d_state, bool update, float* d, float p, int N, int n) {
  curandState state=d_state[blockIdx.x*KERNELBLOCKSIZE+threadIdx.x];
  for (int i=blockIdx.x;i<N;i+=CURANDBLOCKS) {
    for (int j=i*n+threadIdx.x;j<(i+1)*n;j+=KERNELBLOCKSIZE) {
      d[j]=(curand_uniform(&state)<p)?0:d[j];
    }
  }
  if (update)  d_state[blockIdx.x*KERNELBLOCKSIZE+threadIdx.x]=state; //Only update after applying dropout to the derivatives.
}
class curandDropout {
public:
  curandState *d_state;
  curandDropout() {
    cudaMalloc((void**)&d_state,CURANDBLOCKS*KERNELBLOCKSIZE*sizeof(curandState));
    dCurandInit<<<CURANDBLOCKS,KERNELBLOCKSIZE>>>(d_state);
  }
  void drop(float* d, float p, int N, int n) {
    dCurandBernoulliDrop<<<CURANDBLOCKS,KERNELBLOCKSIZE>>>(d_state,false,d,p,N,n);
  }
  void dropD(float* d, float p, int N, int n) {
    dCurandBernoulliDrop<<<CURANDBLOCKS,KERNELBLOCKSIZE>>>(d_state,true,d,p,N,n);
  }
};
///////////////////////////////////////////////////////////////////////////////////////////////////

class SampleWiseDropoutLayer : public Layer {
private:
  RNG rng;
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> mW; //momentum
  vectorCUDA<float> dW; //For backprop
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> mB; //momentum
  vectorCUDA<float> dB; //For backprop
  ActivationFunction fn;
public:
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  curandDropout cd;
  SampleWiseDropoutLayer(int nFeaturesIn, int nFeaturesOut,
                         float dropout=0,ActivationFunction fn=NOSIGMOID) :
    nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut),
    dropout(dropout), fn(fn) {
    float scale=0;
    if (fn!=SOFTMAX)
      scale=powf(nFeaturesIn,-0.5);
    W.resize (nFeaturesIn*nFeaturesOut); W.setUniform(-scale,scale);
    mW.resize (nFeaturesIn*nFeaturesOut); mW.setZero();
    B.resize (nFeaturesOut); B.setZero();
    mB.resize (nFeaturesOut); mB.setZero();

  }
  void forwards
  (BatchInterface &input,
   BatchInterface &output) {
    assert(input.nFeatures==nFeaturesIn);
    assert(input.featuresPresent.size()==nFeaturesIn);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    output.featuresPresent.hVector()=range(nFeaturesOut);
    output.features.resize(output.batchSize*nFeaturesOut);
    replicateArray(B.dPtr(), output.features.dPtr(), output.batchSize, nFeaturesOut);

    if (input.type==TRAINBATCH) {
      output.dfeatures.resize(output.batchSize*nFeaturesOut);
      cd.drop(input.features.dPtr(),dropout,input.batchSize,nFeaturesIn);
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                    output.batchSize, nFeaturesIn, nFeaturesOut,
                                    1.0f, 1.0f,__FILE__,__LINE__);

    } else {
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                    output.batchSize, nFeaturesIn, nFeaturesOut,
                                    1.0f-dropout, 1.0f,__FILE__,__LINE__);
    }
    cudaCheckError();
    applySigmoid(output, output, fn);
  }
  void backwards(BatchInterface &input,
                 BatchInterface &output,
                 float learningRate=0.1) {
    applySigmoidBackProp(output, output, fn);

    dW.resize(nFeaturesIn*nFeaturesOut);
    dB.resize(nFeaturesOut);

    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   input.features.dPtr(), output.dfeatures.dPtr(), dW.dPtr(),
                                   nFeaturesIn, output.batchSize, nFeaturesOut,
                                   1.0, 0.0);
    dB.setZero();
    columnSum(output.dfeatures.dPtr(), dB.dPtr(), output.batchSize, nFeaturesOut);
    cudaCheckError();

    if (input.dfeatures.size()>0)
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                     output.dfeatures.dPtr(), W.dPtr(), input.dfeatures.dPtr(),
                                     output.batchSize,nFeaturesOut,nFeaturesIn,
                                     1.0, 0.0);
    cudaCheckError();

    dGradientDescentNAG<<<nFeaturesIn,KERNELBLOCKSIZE>>>
      (dW.dPtr(), mW.dPtr(), W.dPtr(),  nFeaturesOut, learningRate);
    dGradientDescentNAG<<<1,KERNELBLOCKSIZE>>>
      (dB.dPtr(), mB.dPtr(), B.dPtr(), nFeaturesOut, learningRate);
    cudaCheckError();
    if (input.dfeatures.size()>0)
      cd.dropD(input.dfeatures.dPtr(),dropout,input.batchSize,nFeaturesIn);
  }
  void loadWeightsFromStream(ifstream &f) {
    f.read((char*)&W.hVector()[0],sizeof(float)*W.size());
    f.read((char*)&B.hVector()[0],sizeof(float)*B.size());
  };
  void putWeightsToStream(ofstream &f)  {
    f.write((char*)&W.hVector()[0],sizeof(float)*W.size());
    f.write((char*)&B.hVector()[0],sizeof(float)*B.size());
  };
};
