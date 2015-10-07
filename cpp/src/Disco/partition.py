
def main():
    nPart = 8
    name = 'rcv1_train.binary'
    f = open('%s.4/%s' %(name, name),'rU')

    dic = {}
    b = []
    index = []
    for i in range(nPart):
        index.append([])
    for line in f:
        listofwords = line.split(" ")
        for word in listofwords[1: -1]:
            key =  word.split(":")
            key = int(key[0])
            if key in dic:
                dic[key] += 1
            else:
                dic[key] = 1
    nFeatures = max(dic.items())
    for i in range(nFeatures[0] - 1):
        if i+1 not in dic:
            dic[i+1] = 0

    dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)

    for i in range(nFeatures[0]):
        j = i % nPart
        index[j].append(dic[i][0])

    output = [open('%s.4/%s_%d_%d' % (name, name, nPart, i-1), 'w') for i in range(1, nPart + 1)]  
    f = open('%s.4/%s' %(name, name),'rU')
    no = 0
    for line in f:
        listofwords = line.split(" ")
        linePar = []
        no += 1
        print no
        for j in range(nPart):
           linePar.append([])
        for word in listofwords[1:-1]:
            pair = word.split(":")
            key = int(pair[0])
            for i in range(nPart):
                if key in index[i]:
                    newword = '%d:%s' %(index[i].index(key)+1, pair[1])
                    linePar[i].append(newword)
        for i in range(nPart):
            output[i].write('%2s ' %listofwords[0])
            output[i].write(" ".join(linePar[i]))
            output[i].write("\n")


    f.close()
    for fh in output:
        fh.close()
    
    
    
    
if __name__ == '__main__':
    main()