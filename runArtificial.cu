#include "dropout.h"
#include "readArtificialDataset.h"

int epoch=0;
int cudaDevice=-1;

int main() {
  Dataset trainSet=ArtificialTrainSet();
  Dataset testSet=ArtificialTestSet();

  int batchSize=100;
  trainSet.summary();
  testSet.summary();

  //SimpleNet ann(trainSet.nFeatures, 1000, 3, trainSet.nClasses, RELU, cudaDevice);
  BatchWiseDropoutNet ann(trainSet.nFeatures, 1000, 3, trainSet.nClasses, RELU, 0.5, 0.5, cudaDevice);
  //SampleWiseDropoutNet ann(trainSet.nFeatures, 1000, 3, trainSet.nClasses, RELU, 0.5, 0.5, cudaDevice);

  for (epoch++;epoch<=300;epoch++) {
    cout <<"epoch: " << epoch << " " << flush;
    trainSet.shuffle();
    iterate(ann, trainSet, batchSize,0.001);
    iterate(ann, testSet,  batchSize/4);
  }
}
