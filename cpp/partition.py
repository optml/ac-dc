
def main():
    exp = [4,8,16,32,64]
    name = 'eps'
    for ee in range(5):
        nPart = exp[ee]
        f = open('data/%s.4/%s' %(name, name),'rU')

        dic = {}
        b = []
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
        index = range(nFeatures[0])
        for i in range(nFeatures[0]):
            if i+1 not in dic:
                dic[i+1] = 0

        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        for i in range(nFeatures[0]):
            j = i % nPart
            index[dic[i][0]-1] = j
        
        subIndex = []
        count = []
        for i in range(nPart):
            count.append(0)
        for i in range(nFeatures[0]): 
            count[index[i]] += 1
            subIndex.append(count[index[i]])

        output = [open('data/%s.4/%s_%d_%d' % (name, name, nPart, i-1), 'w') for i in range(1, nPart + 1)]  
        reorder = open('data/%s.4/%s_reorder' % (name, name), 'w')
        f = open('data/%s.4/%s' %(name, name),'rU')
        no = 0
        for line in f:
            listofwords = line.split(" ")
            linePar = []
            reorderPar = []
            no += 1
            #print no
            for j in range(nPart):
                linePar.append([])
            for word in listofwords[1:-1]:
                pair = word.split(":")
                col = int(pair[0])
                newword = '%s:%s' % (subIndex[col-1], pair[1])
                linePar[index[col-1]].append(newword)
                pre = index[col-1]
                accum = 0
                for pp in range(pre):
                    accum += count[pp]
                newwordForReorder = '%s:%s' % (subIndex[col-1] + accum, pair[1])
                reorderPar.append(newwordForReorder)
            for i in range(nPart):
                output[i].write('%s ' %listofwords[0])
                output[i].write(" ".join(linePar[i]))
                output[i].write("\n")
            reorder.write('%s ' %listofwords[0])
            reorder.write(" ".join(reorderPar))
            reorder.write("\n")

        print nPart
        f.close()
        for fh in output:
            fh.close()
    
    
    
    
if __name__ == '__main__':
    main()
