## this code, supplet to submit_jobs_v0.1.py, improves the data formating for matlab analysis
from numpy import *
import sys,os
import time#, pylab
start_time = time.time()


#random.seed()
print('give loading time')
t_loading = 10.
print('Relaxation coefficient')
Dconst=1e-7 # set to 0 to turn off the viscoplasticity channel

print('elastic loading rate:')
if len(sys.argv)!=3:
    print('input frequency realization-number')

N=64
omeg=float(sys.argv[1])
num_realization=int(sys.argv[2])
foldernam = sys.argv[3]
mainfold = sys.argv[4]
k=.05 # machine response
Const=13000.*8./35. # interaction (should be scaled by thrs) thrs should be in the same order of Const
#Const =0.
# Const = 0, I should get 0.67
# Const -> Inf, I should get 1
# Const fixed, but yield stress not fixed

os.chdir(foldernam)
########################################################
def Fext(t,om,J):
    return J*cos(om*t)
def take_random(N):
    #flat part of threshold randomness
    rand_normal = random.uniform(0., 3.,(N,N))
    return rand_normal

dt=1e-2 ## 30s/3000sampling = 1/100 s/sampling = 100 Hz, set limit for the largest omeg
t_actual=0
G=10.
Jamp=30.

dstress_ps = 500. # loading maximum stress
ss=dstress_ps/t_loading*dt
tt0 = int(dstress_ps/ss)
thrs=400.*take_random(N)

genfile='config'+str(num_realization)

#Initialize
strain_arr=zeros((N,N))
stress_arr=strain_arr.copy()

#Euler-solve with just external field stress = F, strain = stress/G 
#Define time then loop over
#Time = 10 periods: 2pi/omega * 10
#t_total=num_cycles*2*pi/omeg

shape=[N,N]
kshape=[N,N]
kx=array(fromfunction(lambda x,y: -pi + 2 * pi * x / float(N) , kshape))
ky=array(fromfunction(lambda x,y: -pi + 2 * pi * y / float(N) , kshape))
kSqrt=sqrt((kx**2+ky**2)+1.0e-16)
kSq=(kx**2+ky**2)+1.0e-16
kSqSq=(kx**2+ky**2)**2+1.0e-16


alpha=-0.05 ##smoothening term, diffusion term of the stress (remove large fluctuation), set it to zero
def calculate_stress_int_longrange(strain):
    Kr = fft.fft2(strain)    
    Ktau = fft.fftshift( (alpha*kSq - Const*kx**2*ky**2/kSqSq)*fft.fftshift(Kr))
    Ktau[0,0]=0.0 # imposed to avoid confusion
    return real(fft.ifft2(Ktau))
def calculate_stress_intMF(strain):
    #return -Const*(average(strain.flatten())-strain)
    return + Const*(average(strain.flatten())-strain)

def find_RHS(strain,forces_thr, stress_ext):
    #finalRHS=calculate_stress_intMF(strain) - forces_thr + stress_ext
    finalRHS=-forces_thr+stress_ext + calculate_stress_int_longrange(strain) - k*average(strain.flatten())
    return finalRHS
def BuildAv(stress_arr,strain_pl,thrs,sext):
    s=find_RHS(strain_pl,thrs,sext)
    avCOND=(s>0) # whenever stress > thrs, avalanche generation
    shell_size=sum(avCOND.flatten())
    avsize=shell_size # how much in total stress > thrs
    avstrainsize=0.
    while shell_size > 0:
        dstrain=avCOND * 1.0 * random.random((N,N))
        strain_pl+=dstrain
        s=find_RHS(strain_pl,thrs,sext)
        avCOND=(s>0)
        shell_size=sum(avCOND.flatten())
        avsize+=shell_size
        avstrainsize+=sum(dstrain.flatten())
    thrs3=400.*take_random(N)
    thrs2=thrs3*avCOND + thrs*(1-avCOND)
    return strain_pl,s,avsize,avstrainsize,thrs2


######################## On the fly testing ##################
fxiaoyue=open(genfile+'_Time'+'.txt','w')

data=[]
sext=0.
strain_pl=zeros((N,N))
thrs0=zeros((N,N))
avnum=0
for i in range(int(t_loading/dt)):
    print(i)
    sext+=ss
    stress_arr=find_RHS(strain_pl,thrs,sext)
    strain_pl,stress_arr,avsize,avstrainsize,thrs=BuildAv(stress_arr,strain_pl,thrs,sext)
    strain_pl+=Dconst*find_RHS(strain_pl,thrs0,sext)
    strain_el_arr=stress_arr/G
    strain_arr=strain_el_arr+strain_pl
    strain_av=average(strain_arr.flatten())
    stress_av=average(stress_arr.flatten())
    if i==0:
        srv0=stress_arr
        srn0=strain_arr
        srn00 = average(strain_arr.flatten())
    
    t_actual+=dt
    #   save data to txt files
#    savetxt('data'+str(i)+'.txt', strain_arr-srn0, fmt='%-8.4f')

    data.append((strain_arr-srn0).flatten())
    fxiaoyue.writelines(\
        format(t_actual,'.4f')+' '+\
        format(sext,'.4f')+' '+\
        format(strain_av-srn00,'.4f')+' \n')

fxiaoyue.close()
savetxt('64_data' + str(random.randint(0, 1000)) + '.txt', array(data), fmt='%-8.4f')
print("--- %s seconds ---" % (time.time() - start_time))
os.chdir(mainfold)

