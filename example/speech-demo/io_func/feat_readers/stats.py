import numpy

class _StreamVariance(object):

    def __init__(self,nCols):
        self.n    = 0;
        self.mean = numpy.zeros(nCols)
        self.M2   = numpy.zeros(nCols)

    def AddX(self,value):
        # do not operate in the same way when the input is an 1
        # dimension array or a 2 dimension array.  Maybe there is
        # a better way to handle that
        if len(value.shape) == 2:
            for x in value:
                self.n     = self.n+1
                delta      = x-self.mean
                self.mean  = self.mean+delta/self.n
                self.M2    = self.M2+delta*(x-self.mean)
        elif len(value.shape) == 1:
            self.n     = self.n+1
            delta      = value-self.mean
            self.mean  = self.mean+delta/self.n
            self.M2    = self.M2+delta*(value-self.mean)
        else:
            msg = 'Only 1D and 2D array are supported'
            raise Exception(msg)

    def GetMean(self):
        return self.mean

    def GetVariance(self):
        return self.M2/(self.n-1)

    def GetInvStandardDeviation(self):
        return 1.0/(numpy.sqrt(self.M2/(self.n-1)))

    def GetNumberOfSamples(self):
        return self.n

class FeatureStats(object):

    def __init__(self):
        self.mean           