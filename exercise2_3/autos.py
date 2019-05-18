
m_begin = 0
comb = []
dif_comb = []
four_door_price = []

with open('imports-85.data', 'r') as f:
    for line in f:
        line = line.split(',')
        if line[2].startswith('m'):
            m_begin += 1
        each_comb = line[3], line[4], line[5], line[6], line[7], line[8]
        if line[3] == '?'or line[4] == "?" or line[5] =='?' or line[6] == '?' or line[7]=='?' or line[8] =="?":
            print('dont use this one')
            continue
        comb.append(list(each_comb))

        if line[5] == 'four' and line[25].rstrip() != '?':
            four_door_price.append(int(line[25].rstrip()))
    for each in comb:
        if each not in dif_comb:
            dif_comb.append(each)
    print(len(dif_comb))
    four_door_price.sort()
    all_price = 0
    for every_price in four_door_price:
        all_price += int(every_price)
    length = len(four_door_price)
    print(length)
    med = 0
    if length % 2 == 0:
        med = (four_door_price[length/2]+four_door_price[length/2-1])/2
    else:
        med = four_door_price[int(length/2)]
    print('1.number of beginning with m is '+str(m_begin))
    print('2.number of different combinations is '+str(len(dif_comb)))
    print('3.average price of four door is '+str(all_price/len(four_door_price)))
    print('3.median price is '+str(med))
