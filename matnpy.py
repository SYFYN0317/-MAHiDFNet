from scipy import io
import numpy as np

mat = np.load('D:\LNNU\FAC0810\FAC0810\index.npy')
#
io.savemat('index.mat', {'index': mat})


# npy = io.loadmat('F:\科研\HRWN-master\HRWN-masterYY\HRWN-master\index.mat')
#
# data=npy['index']
#
# numpy_data=np.transpose(data)
#
# np.save('index.npy',numpy_data)