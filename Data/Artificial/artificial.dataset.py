import numpy
numpy.random.seed(31415927)
def randomWalk(n,l):
    """Random walk in the hypercube {0,1}^n with nearest neighbor edges. Walk of length l-1, so l points in total."""
    a=numpy.zeros((l,n),dtype="float32")
    x=numpy.random.binomial(1,0.5,n)
    a[0]=x
    for j in xrange(1,l):
        i=numpy.random.randint(0,n)
        x[i]=1-x[i]
        a[j]=x
    return a
def randomWalkSampleNoisy(rw,N):
    a=rw[numpy.random.randint(0,rw.shape[0],N)]
    b=numpy.random.binomial(1,0.4,(N,rw.shape[1]))
    return (a+b)%2

nTrain=100000
nTest=10000
dims=1000
walkLengths=1000
nWalksPerClass=1
nClasses=100
fTrain=open("artificial.train.data","wb")
fTest=open("artificial.test.data","wb")
for cl in range(nClasses):
    for w in range(nWalksPerClass):
        rw=randomWalk(dims,walkLengths)
        s=randomWalkSampleNoisy(rw,nTrain/nWalksPerClass/nClasses)
        for i in range(s.shape[0]):
            numpy.array([cl],dtype="uint8").tofile(fTrain)
            s[i].astype("uint8").tofile(fTrain)
        s=randomWalkSampleNoisy(rw,nTest/nWalksPerClass/nClasses)
        for i in range(s.shape[0]):
            numpy.array([cl],dtype="uint8").tofile(fTest)
            s[i].astype("uint8").tofile(fTest)
fTrain.close()
fTest.close()
