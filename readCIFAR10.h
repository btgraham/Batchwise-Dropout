void readCIFAR10File(vector<Datum*> &characters, const char* filename) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);
  }
  unsigned char label;
  while (file.read((char*)&label,1)) {
    vectorDatum* character = new vectorDatum(3072,label);
    unsigned char bitmap[3072];
    file.read((char*)bitmap,3072);
    for (int x=0;x<3072;x++) {
      character->features[x]=bitmap[x]/127.5-1;
    }
    characters.push_back(character);
  }
  file.close();
}
Dataset Cifar10TrainSet() {
  Dataset dataset;
  dataset.name="CIFAR-10 train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3072;
  dataset.nClasses=10;
  char filenameFormat[]="Data/CIFAR10/data_batch_%d.bin";
  char filename[100];
  for(int fileNumber=1;fileNumber<=5;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readCIFAR10File(dataset.samples,filename);
  }
  return dataset;
}
Dataset Cifar10TestSet() {
  Dataset dataset;
  dataset.name="CIFAR-10 test set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3072;
  dataset.nClasses=10;
  char filenameTest[]="Data/CIFAR10/test_batch.bin";
  readCIFAR10File(dataset.samples,filenameTest);
  return dataset;
}
