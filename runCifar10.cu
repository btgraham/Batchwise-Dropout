#include "dropout.h"
#include "readCIFAR10.h"

int cudaDevice=-1;

int main() {
  string baseName="Cifar10";

  Dataset trainSet=Cifar10TrainSet();
  Dataset testSet=Cifar10TestSet();

  int batchSize=100;
  trainSet.summary();
  testSet.summary();

  //SimpleNet ann(trainSet.nFeatures, 1000, 3, trainSet.nClasses, RELU, cudaDevice);
  BatchWiseDropoutNet ann(trainSet.nFeatures, 1000, 3, trainSet.nClasses, RELU, 0.2, 0.5, cudaDevice);
  //SampleWiseDropoutNet ann(trainSet.nFeatures,4000, 3, trainSet.nClasses, RELU, 0.2, 0.5, cudaDevice);
  for (int epoch=1;epoch<=1000;epoch++) {
    cout <<"epoch: " << epoch <<" " << flush;
    trainSet.shuffle();
    iterate(ann, trainSet, batchSize,0.001);
    if (epoch%10==0) 
      iterate(ann, testSet,  batchSize/4);
    
  }
}
