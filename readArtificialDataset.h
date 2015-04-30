static void loadData(string filename, vector<Datum*> &characters, int n) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  unsigned char data[1001];
  for (int i=0;i<n;i++) {
    f.read((char *)data,1001);
    vectorDatum* character = new vectorDatum(1000,data[0]);
    for (int j=0;j<1000;j++)
      character->features[j]=data[j+1]*2-1;
    characters.push_back(character);
  }
}

Dataset ArtificialTrainSet() {
  Dataset dataset;
  dataset.name="Artificial train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=1000;
  dataset.nClasses=100;
  string train("Data/Artificial/artificial.train.data");
  loadData(train, dataset.samples,100000);
  return dataset;
}
Dataset ArtificialTestSet() {
  Dataset dataset;
  dataset.name="Artificial test set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=1000;
  dataset.nClasses=100;
  string train("Data/Artificial/artificial.test.data");
  loadData(train, dataset.samples,10000);
  return dataset;
}
