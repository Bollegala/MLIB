#! /usr/bin/python
#! -*- coding: utf-8 -*-

######################################################################
# Example showing how to implement logistic regression
# using stochastic gradient decent.
#
# 2012/05/03
# Danushka Bollegala
######################################################################

import sys,math,time
sys.path.append("../..")

import numpy
import pprint
import matplotlib.pyplot as plt

from MLIB.utils.data import  convertSEQFileToArray, mapTestVects



class SGD:

    def __init__(self):
        """
        Initialiing parameters.
        """
        self.eta0 = 0.1
        self.lmda = 1.0
        self.totalIterations = 10
        self.alpha = 0.9
        pass
    

    def train(self, xTrain, yTrain):
        """
        Perform training with SGD.
        """
        N = len(yTrain)
        M = len(xTrain[0])
        b = 0
        w = numpy.ones(M, dtype=numpy.float64)
        count = 0
        PP = pprint.PrettyPrinter(indent=4)

        counts = []
        llscores = []
        accs = []
        etas = []
        bvals = []
        
        for i in range(self.totalIterations):
            print "Iteration = %d" % i
            for j in range(N):
                count += 1
                
                x = xTrain[j]
                y = yTrain[j]

                # Exponential Learning Rate.
                #eta = self.eta0 * numpy.power(self.alpha, (float(count) / float(N *self.totalIterations)))
                eta = self.eta0 / (1.0 + (float(count) / float(self.totalIterations * N)))
                
                score = b + numpy.inner(w, x)
                a = self.sigmoid(score)

                factor = (1.0 - ((2.0 * eta * self.lmda) / float(N)))
                w = (factor * w) + (eta * (y - a) * x)

                b = b + (eta * (y - a))
                pass
            counts.append(i)
            bvals.append(b)
            etas.append(eta)
            res = self.getPerformance(yTrain, xTrain, w, b)
            accs.append(res['Acc'])
            llscores.append(res['ll'])
            pass        

        plt.figure()
        plt.plot(counts, accs, 'r+-')
        plt.title("Accuracy")

        plt.figure()
        plt.plot(counts, llscores, 'b+-')
        plt.title("Log Likelihood")

        plt.figure()
        plt.plot(counts, etas, 'g+-')
        plt.title("Eta value")

        plt.figure()
        plt.plot(counts, bvals, 'm+-')
        plt.title("Bias term")
        
        plt.show()
        return (w, b)
            

    def getPerformance(self, labels, vects, w, b):
        """
        Evaluate the performance.
        """
        TP = FP = TN = FN = 0
        lltot = 0
        
        for i in range(len(vects)):
            t, p, lp = self.predict(vects[i], w, b)
            if labels[i] == 1 and t == 1:
                TP += 1
            elif labels[i] == 1 and t == 0:
                FN += 1
            elif labels[i] == 0 and t == 1:
                FP += 1
            elif labels[i] == 0 and t == 0:
                TN += 1
            else:
                sys.stderr.write("%d %d Invalid label or target!\n" % (labels[i], t))
                raise ValueError
            lltot += lp
            
        res = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN,
               'P+': 0 if (TP + FP) == 0 else float(TP) / float(TP + FP),
               'R+': 0 if (TP + FN) == 0 else float(TP) / float(TP + FN),
               'P-': 0 if (FN + TN) == 0 else float(TN) / float(FN + TN),
               'R-': 0 if (FP + TN) == 0 else float(TN) / float(FP + TN)}
        res['F+'] = 0 if (res['P+'] * res['R+']) == 0 else (2 * res['P+'] * res['R+']) / (res['P+'] + res['R+'])
        res['F-'] = 0 if (res['P-'] * res['R-']) == 0 else (2 * res['P-'] * res['R-']) / (res['P-'] + res['R-'])
        res['Acc'] = float(TP + TN) / float(TP + TN + FP + FN)
        res['ll'] = lltot
        return res
    

    def predict(self, x, w, b):
        """
        Predicts the label and class conditional probability.
        """
        score = b + numpy.inner(w, x)
        #print score
        if score > 0:
            t = 1
        else:
            t = 0
        p = self.sigmoid(score)
        if score < -700:
            lp = score
        else:
            lp = numpy.log(p)
        return (t, p, lp)
    

    def sigmoid(self, x):
        if x < -700:
            return 0
        val = 1.0 / (1.0 + math.exp(-x))
        return val

    pass


if __name__ == "__main__":

    def loadData(fname):
        """
        Read the feature vector file.
        Convert labels from {-1,+1} to {0,1}.
        """
        labels, vects =  convertSEQFileToArray(fname)
        convLabels = [(y + 1) / 2  for y in labels]
        return (convLabels, vects)
    

    def test():
        trainFileName = "../../FTL/data/lukemia/train.scaled"
        testFileName = "../../FTL/data/lukemia/test.scaled"
        logreg = SGD()
        yTrain, xTrain = loadData(trainFileName)

        d = len(xTrain[0])
        yTest, xTest = loadData(testFileName)
        for i in range(len(xTest)):
            xTest[i] = mapTestVects(xTest[i], d)
        
        w, b = logreg.train(xTrain, yTrain)

        PP = pprint.PrettyPrinter(indent=4)
        print "\n\n============== Train Accuracy ==============="
        trainRes = logreg.getPerformance(yTrain, xTrain, w, b)
        PP.pprint(trainRes)
        
        print "\n\n============== Test Accuracy ================"
        testRes = logreg.getPerformance(yTest, xTest, w, b)
        PP.pprint(testRes)        
        pass

    test()
    

        
        

