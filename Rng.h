boost::mutex RNGseedGeneratorMutex;
boost::mt19937 RNGseedGenerator;

class RNG {
  timespec ts;
public:
  boost::mt19937 gen;

  RNG() {
    clock_gettime(CLOCK_REALTIME, &ts);
    RNGseedGeneratorMutex.lock();
    gen.seed(RNGseedGenerator()+ts.tv_nsec);
    RNGseedGeneratorMutex.unlock();
  }
  int randint(int n) {
    if (n==0) return 0;
    else return gen()%n;
  }
  float uniform(float a=0, float b=1) {
    unsigned int k=gen();
    return a+(b-a)*k/4294967296.0;
  }
  float normal(float mean=0, float sd=1) {
    boost::normal_distribution<> nd(mean, sd);
    boost::variate_generator<boost::mt19937&,
                             boost::normal_distribution<> > var_nor(gen, nd);
    return mean+sd*var_nor();
  }
  int bernoulli(float p) {
    if (uniform()<p)
      return 1;
    else
      return 0;
  }
  template <typename T>
  int index(std::vector<T> &v) {
    if (v.size()==0) std::cout << "RNG::index called for empty std::vector!\n";
    return gen()%v.size();
  }
  std::vector<int> zNchooseM(int n, int m) {
    std::vector<int> ret;
    for(int i=0;i<n;i++)
      if (uniform()*n<m) ret.push_back(i);
    return ret;
  }
  std::vector<int> NchooseM(int n, int m) {
    std::vector<int> ret(m);
    int ctr=m;
    for(int i=0;i<n;i++)
      if (uniform()<ctr*1.0/(n-i)) ret[m-ctr--]=i;
    return ret;
  }
  std::vector<int> permutation(int n) {
    std::vector<int> ret;
    for (int i=0;i<n;i++) ret.push_back(i);
    random_shuffle ( ret.begin(), ret.end());
    return ret;
  }
};
