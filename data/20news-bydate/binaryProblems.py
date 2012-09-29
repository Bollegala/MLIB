"""
Create binary classification problems as mentioned in the Confidence-Weighted
Linear Classification paper by Derdze et al.

comp.sys.ibm.pc.hardware 4
comp.sys.mac.hardware 5

sci.electronics 13
sci.med 14

talk.politics.guns 17
talk.politics.mideast 18

Danushka Bollegala.
2012/05/22.
"""


def createDocFile(inputFileName, outputFileName):
    """
    Create a single file in the following format.
    docid wid:count wid:count ...
    wids are sorted in the ascending order.
    """
    fin = open(inputFileName)
    fout = open(outputFileName, "w")
    old_docid = 1
    line = fin.readline()
    fout.write("1 ")
    while line:
        p = line.strip().split()
        docid = int(p[0])
        if docid != old_docid:
            fout.write("\n%d " % docid)
            old_docid = docid
        wid = int(p[1])
        count = int(p[2])
        fout.write("%d:%d " % (wid, count))
        line = fin.readline()
    fout.write("\n")
    fin.close()
    fout.close()   
    pass


def createBinary(posLabel, negLabel, domain, mode):
    """
    creates train instances for each binary problem.
    """
    labelFile = open("./original/%s.label" % mode)
    itemFile = open("./original/%s.all" % mode)
    instanceFile = open("./%s/%s-nonscaled.%s" % (domain, domain, mode), "w")

    labels = [int(x.strip()) for x in labelFile.readlines()]
    vects = [x.strip() for x in itemFile.readlines()]
    labelFile.close()
    itemFile.close()

    n = len(labels)
    posCount = negCount = 0
    for i in range(0,n):
        p = vects[i].split()
        if labels[i] == posLabel:
            # This is a positive instance.
            instanceFile.write("1 %s\n" % " ".join(p[1:]))
            posCount += 1
        elif labels[i] == negLabel:
            # This is a negative instance.
            instanceFile.write("-1 %s\n" % " ".join(p[1:]))
            negCount += 1
    instanceFile.close()
    print "Positive Instances =", posCount
    print "Negative Instances =", negCount
    pass


def main():
    # Create train and test instance files.
    #createDocFile("./original/train.data", "./original/train.all")
    #createDocFile("./original/test.data", "./original/test.all")

    # Create binary classification instances.
    createBinary(17, 18, "talk", "test")    
    pass


if __name__ == "__main__":
    main()

