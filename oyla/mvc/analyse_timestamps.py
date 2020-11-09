##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Version:           2.0 20.10.2019
# Description                                                                                    #
#        main  script for  analysing timestamps for requesting (pre), getting (post) and plot (updating) frames
#  For multiple chips controller.py has to be run in debug mode so as to store indiviual camera/chip data
# red, green is a chip (1), blue, black is the other chip (2). dot is thread1 and square is thread2
# yellow is update view (plot)
# only one input argument -- data_dir (input where mat files are stored); 
##################################################################################################
import scipy.io
import glob
import matplotlib.pyplot as plt
import sys
import os
data_dir = sys.argv[1]

files = glob.glob(os.path.join(data_dir,'data_?_*.mat'))

plt.plot(hold = 'on')
single_chip = True
for f in files:
    A = scipy.io.loadmat(f)
    frame_number = int(A['data'][0][2][0][2][0][0])
    chip_number = A['data'][0][3][0]
    thread = A['data'][0][5][0]
    pre = A['data'][0][1][0][0]
    post = A['data'][0][0][0][0]
    if len(A['data'][0])> 6:
        update = A['data'][0][6][0]
    else:
        single_chip = False
    if chip_number == '1':
        
        if thread == '1T':
            plt.plot(pre,frame_number,'r.')
            plt.plot(post,frame_number,'g.')
        else:
            plt.plot(pre,frame_number,'rs')
            plt.plot(post,frame_number,'gs')
    else:
        single_chip = False
        if thread == '1T':
            plt.plot(pre,frame_number,'b.')
            plt.plot(post,frame_number,'k.')
        else:
            plt.plot(pre,frame_number,'bs')
            plt.plot(post,frame_number,'ks')
    if single_chip:
        plt.plot(update,frame_number,'yp')

if not single_chip:
    files = glob.glob(os.path.join(data_dir,'data_stitched_*.mat'))
    for f in files:
        A = scipy.io.loadmat(f)
        frame_number = int(A['data'][0][2][0][2][0][0])
        update = A['data'][0][6][0]

        plt.plot(update,frame_number,'yp')
plt.show()
