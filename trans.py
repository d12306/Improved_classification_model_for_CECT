import numpy as np
import pickle

center = open('/shared/xiangruz/classification/SHREC/test_data/centers.pickle','rb')
centers = pickle.load(center,encoding = 'latin1')

label = np.load('./pids.npy')
f1 = open('./result.txt','w')
print(len(label))
for i in range(len(label)):
    if label[i][0]!='none':
    	f1.write(str(label[i][0])+' '+str(centers[i][0])+' '+str(centers[i][1])+' '+str(centers[i][2])+'\n')
    else:
   		print ('fffff')

f1.close()
