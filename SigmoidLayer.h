//   _____ _                       _     _
//  / ____(_)                     (_)   | |
// | (___  _  __ _ _ __ ___   ___  _  __| |___
//  \___ \| |/ _` | '_ ` _ \ / _ \| |/ _` / __|
//  ____) | | (_| | | | | | | (_) | | (_| \__ \
// |_____/|_|\__, |_| |_| |_|\___/|_|\__,_|___/
//            __/ |
//           |___/

enum ActivationFunction             {NOSIGMOID, RELU,   VLEAKYRELU,   LEAKYRELU, LOGISTIC, TANH,  SOFTMAX};
const char *sigmoidNames[] ={ "*"       , "ReLU", "VeryLeakyReLU", "LeakyReLU", "LOGISTIC", "Tanh", "Softmax Classification"};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidLogistic
(float* a, float* b, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=1/(1+expf(-a[j]));
  }
}
void sigmoidLogistic(float* a, float* b, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidLogistic<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropLogistic
(float* a, float* b, float* da, float* db, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=db[j]*b[j]*(1-b[j]);
  }
}
void sigmoidBackpropLogistic(float* a, float* b, float* da, float* db, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidBackpropLogistic<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidTanh
(float* a, float* b, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=tanhf(a[j]);
  }
}
void sigmoidTanh(float* a, float* b, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidTanh<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropTanh
(float* a, float* b, float* da, float* db, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=db[j]*(1+b[j])*(1-b[j]);
  }
}
void sigmoidBackpropTanh(float* a, float* b, float* da, float* db, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidBackpropTanh<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidReLu
(float* a, float* b, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=(a[j]>0)?a[j]:0;
  }
}
void sigmoidReLU(float* a, float* b, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropReLu
(float* a, float* b, float* da, float* db, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=(a[j]>0)?db[j]:0;
  }
}
void sigmoidBackpropReLU(float* a, float* b, float* da, float* db, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidBackpropReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidLeakyReLu
(float* a, float* b, int nOut, float alpha) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=(a[j]>0)?a[j]:(a[j]*alpha);
  }
}
void sigmoidLeakyReLU(float* a, float* b, int count, int nOut, float alpha=0.01) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidLeakyReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut,alpha);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropLeakyReLu
(float* a, float* b, float* da, float* db, int nOut, float alpha) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=(a[j]>0)?db[j]:(db[j]*alpha);
    __syncthreads();
  }
}
void sigmoidBackpropLeakyReLU(float* a, float* b, float* da, float* db, int count, int nOut, float alpha=0.01) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidBackpropLeakyReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut,alpha);
    processed+=batch;
  }
  cudaCheckError();
}




//SOFTMAX only occurs at the top layer;
//derivative contained in calculation of initial d_delta.
__global__ void dSigmoidSoftmax(float* a, float* b, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=NTHREADS) {
    float acc=0.0f;
    float mx=-10000.0f;
    for (int k=0;k<nOut;k++)
      if (a[i*nOut+k]>mx) mx=a[i*nOut+k];
    for (int k=0;k<nOut;k++) {
      b[i*nOut+k]=expf(a[i*nOut+k]-mx); //Subtract row max value for numerical stability.
      acc+=b[i*nOut+k];}
    for (int k=0;k<nOut;k++) {
      b[i*nOut+k]/=acc;
    }
  }
}
__global__ void dSigmoidBackpropSoftmax(float* a, float* b, float* da, float* db, int count, int nOut) {
  for(int i=0; i<count; i++) {
    for (int k=threadIdx.x; k<nOut; k+=NTHREADS) {
      da[i*nOut+k]=db[i*nOut+k];
    }
  }
}
void applySigmoid(BatchInterface& input, BatchInterface& output, ActivationFunction fn) {
  switch(fn) {
  case TANH:
    sigmoidTanh
      (input.features.dPtr(),
       output.features.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case LOGISTIC:
    sigmoidLogistic
      (input.features.dPtr(),
       output.features.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case RELU:
    sigmoidReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case LEAKYRELU:
    sigmoidLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.batchSize,
       output.featuresPresent.size(),
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.batchSize,
       output.featuresPresent.size(),
       0.333);
    break;
  case SOFTMAX:     dSigmoidSoftmax      <<<1,NTHREADS>>> (input.features.dPtr(),output.features.dPtr(),output.batchSize,output.nFeatures);   break;
  }
}

void applySigmoidBackProp(BatchInterface& input, BatchInterface& output, ActivationFunction fn) {
  switch(fn) {
  case TANH:
    sigmoidBackpropTanh
      (input.features.dPtr(),output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case LOGISTIC:
    sigmoidBackpropLogistic
      (input.features.dPtr(),output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case RELU:
    sigmoidBackpropReLU
      (input.features.dPtr(),output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.batchSize,
       output.featuresPresent.size());
    break;
  case LEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.batchSize,
       output.featuresPresent.size(),
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.batchSize,
       output.featuresPresent.size(),
       0.333);
    break;
  case SOFTMAX:
    dSigmoidBackpropSoftmax  <<<1,NTHREADS>>>
      (input.features.dPtr(),output.features.dPtr(), input.dfeatures.dPtr(),output.dfeatures.dPtr(), output.batchSize, output.nFeatures);   break;
  }
}

class SigmoidLayer : public Layer {
public:
  ActivationFunction fn;
  SigmoidLayer(ActivationFunction fn) : fn(fn) {
    cout << sigmoidNames[fn]<<endl;
  };
  void forwards(BatchInterface &input,BatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    output.features.resize(output.batchSize*output.featuresPresent.size());
    if (input.type==TRAINBATCH)
      output.dfeatures.resize(output.batchSize*output.featuresPresent.size());
    applySigmoid(input, output, fn);
  }
  void backwards(BatchInterface &input,
                 BatchInterface &output,
                 float learningRate) {
    applySigmoidBackProp(input, output, fn);
  }
};
