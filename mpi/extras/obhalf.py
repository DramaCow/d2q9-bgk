f = open('obstacles_half.dat', 'w')
for y in range(0,128):
  for x in range(0,256):
    f.write(str(x) + ' ' + str(y) + ' 1\n')
