import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import matplotlib as mpl
import matplotlib.cm as cm

mpl.use('Agg')

###########################################
############### fonctions #################
###########################################

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
            if arr[l][k] == 255:
                f.append(complex(k, l))

    f = np.array(f)
    T = len(f)

    f_cont = [f[0]]
    f_cop = np.delete(np.copy(f), 0)
    for k in range(T-1):                        #return a continous by parts function for the TF

        i = np.argmin(np.absolute(f_cont[-1] - f_cop))
        f_cont.append(f_cop[i])

        f_cop = np.delete(f_cop, i)

    C = []
    for k in range(N):                          #calculation of fourier coefficients
        Cpp, Cmm = inte(f_cont, T, k)
        C.extend((Cpp, Cmm))

    return np.array(C[1:])

###########################################
################# main ####################
###########################################


#######   canny filter   ############

print("Step one : Canny filter")

img_name = 'CutPaste_2022-09-27_23-19-19-330.png'
path = 'img_source/' + img_name
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255 * (gray < 100)                        #binarisation

grayCopy = np.uint8(gray)
slicecanny = cv2.Canny(grayCopy, 0, 255)

print("Canny filter successed\n")

#######   variable   ############

print("Step 2 : calculation of CF")

N = 100
tf = 2*np.pi
pas = 0.01
t = np.arange(0, tf, pas)
C = CF(slicecanny, N, t)
digit = 4

print("CF successed\n")
print("step 3 : img creation")

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

it = 0
OM_tot = []
OM1_tot = []
OM2_tot = []
OM3_tot = []
for i in t:

    fig = plt.figure(figsize = (9,9))
    fig.patch.set_facecolor('black')

    ax = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    """
    cir = plt.Circle((0, 0), np.absolute(C[0]), ec = "orange", fc = "none", zorder = 1e5)
    ax.add_patch(cir)
    plt.plot([0, C[0].real], [0, C[0].imag])
    """

    OM = C[0]
    l = 0
    for k in range(1, 2 * N - 1):

        if k%2 != 0:
            l = l+1

        OM1 = np.copy(OM)
        OM = OM + C[k] * np.exp(1j * i * l * (-1)**(k+1))

        ax.add_patch(plt.Circle((OM1.real, OM1.imag), np.absolute(C[k]), ec = "grey", fc = "none", zorder = 2))
        ax.plot([OM1.real, OM.real], [OM1.imag, OM.imag], color = "white")

        if k<=100:
            ax1.add_patch(plt.Circle((OM1.real, OM1.imag), np.absolute(C[k]), ec = "grey", fc = "none", zorder = 2))
            ax1.plot([OM1.real, OM.real], [OM1.imag, OM.imag], color = "white")
            if k == 100:
                OM1_tot.append(OM)


        if k<=50:
            ax2.add_patch(plt.Circle((OM1.real, OM1.imag), np.absolute(C[k]), ec = "grey", fc = "none", zorder = 2))
            ax2.plot([OM1.real, OM.real], [OM1.imag, OM.imag], color = "white")
            if k == 50:
                OM2_tot.append(OM)

        if k<=20:
            ax3.add_patch(plt.Circle((OM1.real, OM1.imag), np.absolute(C[k]), ec = "grey", fc = "none", zorder = 2))
            ax3.plot([OM1.real, OM.real], [OM1.imag, OM.imag], color = "white")
            if k == 20:
                OM3_tot.append(OM)



    OM_tot.append(OM)
    ax.plot(np.array(OM_tot).real, np.array(OM_tot).imag, "-", color = cm.coolwarm(0), markersize = 15, zorder = 4)
    ax1.plot(np.array(OM1_tot).real, np.array(OM1_tot).imag, "-", color = cm.coolwarm(0), markersize = 15, zorder = 4)
    ax2.plot(np.array(OM2_tot).real, np.array(OM2_tot).imag, "-", color = cm.coolwarm(0), markersize = 15, zorder = 4)
    ax3.plot(np.array(OM3_tot).real, np.array(OM3_tot).imag, "-", color = cm.coolwarm(0), markersize = 15, zorder = 4)





    ax.set_xlim(0, len(slicecanny[0]))
    ax.set_ylim(len(slicecanny), 0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('black')
    ax.set_title("n_max = 200", color = "white")


    ax1.set_xlim(0, len(slicecanny[0]))
    ax1.set_ylim(len(slicecanny), 0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_facecolor('black')
    ax1.set_title("n_max = 100", color = "white")



    ax2.set_xlim(0, len(slicecanny[0]))
    ax2.set_ylim(len(slicecanny), 0)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_facecolor('black')
    ax2.set_title("n_max = 50", color = "white")

    ax3.set_xlim(0, len(slicecanny[0]))
    ax3.set_ylim(len(slicecanny), 0)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_facecolor('black')
    ax3.set_title("n_max = 20", color = "white")



    name_pic = name(int(it), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    ax1.clear()
    plt.close(fig)

    print(i/tf)

    it = it + 1

print("img successed")

# ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"


