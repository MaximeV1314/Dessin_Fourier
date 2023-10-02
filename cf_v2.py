import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl

#mpl.use('Agg')

###########################################
############### fonctions #################
###########################################

digit = 4
def name(i,digit):

    i = str(i)

    while len(i)<digit:
        i = '0'+i

    i = 'img/'+i+'.png'

    return(i)

def inte(f, T, n):

    Cnp = 0
    Cnm = 0
    for i in range(T):
        Cnp = Cnp + f[i] * np.exp(- 2j * np.pi * n * i/T)
        Cnm = Cnm + f[i] * np.exp(2j * np.pi * n * i/T)

    return Cnp/T, Cnm/T

def CF(arr, N, t):

    f = []                                    #detecte all white pixels on the image with a canny filter applied
    for k in range(len(arr[0])):          #and return a complex array with all positions
        for l in range(len(arr)):
            if slicecanny[l][k] == 255:
                f.append(complex(k, l))

    f = np.array(f)
    T = len(f)

    f_cont = [f[0]]
    f_cop = np.delete(np.copy(f), 0)
    for k in range(T-1):                        #return a continous by parts function for the TF

        i = np.argmin(np.absolute(f_cont[-1] - f_cop))
        f_cont.append(f_cop[i])

        f_cop = np.delete(f_cop, i)

    Cp = []
    Cm = []
    for k in range(N):                          #calculation of fourier coefficients
        Cpp, Cmm = inte(f_cont, T, k)
        Cp.append(Cpp)
        Cm.append(Cmm)

    Cp = np.array(Cp)
    Cm = np.array(Cm)
    X = []
    Y = []

    N_list = np.arange(1, N, 1)
    for k in t:
        OM = Cp[0] + np.sum(Cp[1:] * np.exp(1j * k * N_list) + Cm[1:] * np.exp(-1j * k * N_list))
        X.append(OM.real)
        Y.append(OM.imag)

    return X, Y

###########################################
################# main ####################
###########################################

#######   canny filter   ############

img_name = 'noel.png'
path = 'img_source/' + img_name
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255 * (gray < 180)                        #binarisation

grayCopy = np.uint8(gray)
slicecanny = cv2.Canny(grayCopy, 0, 255)

#######   plot   ############

N = 500
tf = 7
pas = 0.01
t = np.arange(0, tf, pas)
X, Y = CF(slicecanny, N, t)

#for i in range(629, 770):
fig = plt.figure(figsize = (9,9))
fig.patch.set_facecolor('black')

ax1 = fig.add_subplot(111)
ax1.set_xlim(0, len(slicecanny[0]))
ax1.set_ylim(len(slicecanny), 0)
ax1.set_aspect('equal', adjustable='box')
ax1.set_facecolor('black')

ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.plot(X, Y, "-")
#, linewidth = 3.5

#name_pic = name(int(i), digit)
#plt.savefig(name_pic, bbox_inches='tight', dpi=300)

#ax1.clear()
#plt.close(fig)


#ax2 = fig.add_subplot(122)
#ax2.imshow(slicecanny, cmap='gray')

plt.show()


