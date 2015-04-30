#include "dropout.h"
#include "readMNIST.h"

int cudaDevice=3;

int main() {
  string baseName="/tmp/mnist";

  Dataset trainSet=MnistTrainSet();
  Dataset testSet=MnistTestSet();

  int batchSize=100;
  trainSet.summary();
  testSet.summary();

  //SimpleNet ann(trainSet.nFeatures, 800, 2, trainSet.nClasses, RELU, cudaDevice);
  BatchWiseDropoutNet ann(trainSet.nFeatures, 800, 2, trainSet.nClasses, RELU, 0.2, 0.5, cudaDevice);
  //SampleWiseDropoutNet ann(trainSet.nFeatures, 800, 2, trainSet.nClasses, RELU, 0.2, 0.5, cudaDevice);

  for (int epoch=1;epoch<=200;epoch++) {
    cout <<"epoch: " << epoch << " " << flush;
    trainSet.shuffle();
    iterate(ann, trainSet, batchSize,0.01*exp(-epoch*0.01));
    if (epoch%10==0)
      iterate(ann, testSet,  batchSize/4);
  }
}
