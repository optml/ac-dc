from math import ceil, floor


def main():
    nPart = 8
    name = 'kdd'

    nnlines = 0
    output = [open('data/%s.4/%s.%d.%d' % (name, name, nPart, i-1), 'w') for i in range(1, nPart + 1)]  
    f = open('data/%s.4/%s' %(name, name),'rU')
    for line in f:
        nnlines += 1

    nnsamplesPerFile = ceil(nnlines / nPart)+1

    count = 0
    f = open('data/%s.4/%s' %(name, name),'rU')
    for line in f:
        i = floor(count / nnsamplesPerFile)
        output[int(i)].write(line)
        count +=1

    f.close()
    for fh in output:
        fh.close()
    
    
    
    
if __name__ == '__main__':
    main()
