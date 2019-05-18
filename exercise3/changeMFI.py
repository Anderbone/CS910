

with open('abalone1.data','w') as w:
    with open('abalone.data', 'r') as f:
        for line in f:
            line = line.split(',')
            if line[0] == 'I':
                line[0] = 0
            else:
                line[0] = 1

            s = ','.join(str(i) for i in line).strip()
            print(s, file=w)