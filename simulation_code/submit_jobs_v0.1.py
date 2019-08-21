## this code improves the data formating for matlab analysis
import sys,os,glob
from subprocess import Popen, PIPE
from scipy import arange
import time
from scipy import *

mainfolder = os.getcwd()
Nprocs=2
rands=[]
#random.seed()
Niter=5 # Number of realizations for each of the omegas
omegas=array([1.])
j=0

for j, omega in enumerate(omegas):
    foldername = 'om'+str(j)
    try:
        os.mkdir(foldername)
    except:
        print('folder exists')
    for i in range(Niter):
        cmd="python PlastDMA_v3.4_SimpleRamp.py "+str(omega)+" "+str(i)+" "+foldername+" "+mainfolder
        print(cmd)
        print(str(i) + 'done')
        if (j+1)%(Nprocs+1)==0:
            os.system(cmd)
        else:
            os.system(cmd+" &")
        j+=1

