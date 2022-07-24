import re
import os
import numpy
import matplotlib.pyplot as plt
LABEL = "OPTIONAL"

extract_loss = re.compile(r"Epoch (\d+) finished, average loss: (\d.\d+)")
print(os.listdir())
arr_loss = []
with open('LM-ResNet-master copy/ResNet20Normal.txt') as f:
    ctr = 0
    for i,line in enumerate(f):
        if i%100==0: print(i)
        match = extract_loss.match(line)
        if match:
            arr_loss.append(match.group(2))
print('over')
nparr_loss = numpy.array(arr_loss)
nparr_loss = nparr_loss.astype(float)
x = numpy.arange(len(nparr_loss))
plt.plot(x,nparr_loss)#, label=label)
print('over2')
plt.show()
print('over3')
#plt.savefig(FIG_FILENAME)