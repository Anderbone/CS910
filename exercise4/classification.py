with open('car1.data','r') as f:
    for line in f:
        line = line.split(',')

        if line[3] == '2' and line[6].strip() == 'acc':
            print('got one')