
with open('car1.data', 'w') as w:
    with open('car.data', 'r') as f:
        for line in f:
            line = line.split(',')
            if line[6].strip() == 'good' or line[6].strip() == 'vgood':
                line[6] = 'acc'

            s = ','.join(str(i) for i in line).strip()
            print(s, file=w)
