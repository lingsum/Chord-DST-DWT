# Guitar chord recogition using dst-dwt feature extraction
# Version: fix

import numpy as np
import math
import os
import pywt
from scipy import signal
from scipy.fftpack import fft,dst
from scipy.io import wavfile

# =============================================================
# Function Definition
# =============================================================
def convolution(x,y):
    z = signal.convolve(x,y)
    lenx = len(x)
    lenz = len(z)
    zcenter = math.ceil(lenz/2)
    zstart = int(zcenter-(lenx/2))
    zout = z[zstart:zstart+lenx]
    return zout

# =============================================================
# Multilevel wavelet decomposition
def wdecmulti(X,wfilter):
    W = pywt.Wavelet(wfilter)
    LPFD = W.dec_lo       # lowpass filter (decomposition)
    HPFD = W.dec_hi       # highpass filter (decomposition)
    YOut = np.array([])
    numdec = int(math.log2(len(X)))
    for k in range(numdec):
        YL0 = convolution(X,LPFD)   # lowpass filtering
        YH0 = convolution(X,HPFD)   # highpass filtering
        YLD = YL0[0::2]     # downsampling
        YHD = YH0[0::2]     # downsampling
        X = YLD;
        if k < numdec-1:
            YOut = np.hstack((YHD,YOut))
        else:
            YOut = np.hstack((YLD,YHD,YOut))
    return YOut

# =============================================================
# Input data - preprocessing -  feature extraction
def xcinput(wav4,frame4,dstcut4,wave4,coef4):
    # 1. Input data
    fs, y0 = wavfile.read(wav4)
    
    # 2. Normalization
    maxy = max(abs(y0))
    y1 = y0/maxy

    # 3. Silence and transition time cutting
    # 3a. Silence time cutting
    th = 0.5                            # Threshold value 0.5
   
    ax = np.where(abs(y1)>th)
    bx = np.array(ax)
    bx = bx.flatten()
    y2 = np.delete(y1,range(bx[0]))
    
    # 3b. Transition time cutting
    bts = np.arange(int(0.2*fs))        # Cutting time 200 ms
    y3 = np.delete(y2, bts)

    # 4. Frame blocking
    y4 = y3[0:frame4]

    # 5. Windowing
    window = signal.windows.hamming(frame4)
    y5 = y4*window

    # 6. FFT
    y6 = abs(fft(y5))
    fftframe=frame4/2
    y6 = y6[0:int(fftframe)]
 
    # 7. DST
    y7 = dst(y6)
    y7 = y7[0:int(fftframe*dstcut4)]
    
    # 8. Multilevel wavelet decomposition
    y8 = wdecmulti(y7,wave4)
    
    # 9. Feature selection
    y9 = y8[0:coef4]
    
    # 10. Normalization
    maxout = max(abs(y9))
    xcout = y9/maxout
    
    return xcout

# =============================================================
# Distance computation using cosine distance
def distcomp(x,y):
    c1 = sum(x*y)
    c2 = (sum(x*x))**0.5
    c3 = (sum(y*y))**0.5
    z = 1 - (c1/((c2*c3)+2.23*10**-16)) # Add very small value to
    return z                            # avoid division by zero

# =============================================================
# Chord classification
def chordclass(db,test,classin):
  
    # Distance computation
    distarray = np.zeros(7)
    for k in range(7):
        distarray[k] = distcomp(test,db[k])
    
    # Output classification
    sortidx = np.argsort(distarray)  # ascending sort
    classout = sortidx[0]            # output class
    chords = np.array(['c','d','e','f','g','a','b'])    # 7 chords  
    chordout=chords[classout]
    
    # Check the correctness of the output
    if classout == classin:
        trueout = 1
    else:
        trueout = 0

    return chordout,trueout

# =============================================================      
# Feature extraction for testing
def xctest(chord3,frame3,dstcut3,wave3,coef3):
    xc2 = np.zeros(shape=(20,coef3))
    m = 1
    for k in range(20):
        path2 = 'd:/Data/Research/Dataset/guitards/g'
        wavpath2 = "".join([path2,chord3,str(m+10), '.wav'])
        wav2 = os.path.abspath(wavpath2)
        wxc2 = xcinput(wav2,frame3,dstcut3,wave3,coef3)
        xc2[k,:] = wxc2
        m = m+1
    return xc2

# =============================================================      
# Feature extraction for database
def xcdb(chord2,frame2,dstcut2,wave2,coef2):
    xc1 = np.zeros(shape=(10,coef2))
    for k in range(10):
        path1 = 'd:/Data/Research/Dataset/guitards/g'
        wavpath1 = "".join([path1,chord2,str(k+1), '.wav'])
        wav1 = os.path.abspath(wavpath1)
        wxc1 = xcinput(wav1,frame2,dstcut2,wave2,coef2)
        xc1[k,:] = wxc1
    db1 = np.mean(xc1,axis=0)      # column averaging
    return db1

# =============================================================
# Database, datatest, and recognition
def datarecog(frame1,dstcut1,wave1,coef1):
    # Database: 10 samples/class
    dc = xcdb('c',frame1,dstcut1,wave1,coef1)
    dd = xcdb('d',frame1,dstcut1,wave1,coef1)
    de = xcdb('e',frame1,dstcut1,wave1,coef1)
    df = xcdb('f',frame1,dstcut1,wave1,coef1)
    dg = xcdb('g',frame1,dstcut1,wave1,coef1)
    da = xcdb('a',frame1,dstcut1,wave1,coef1)
    db = xcdb('b',frame1,dstcut1,wave1,coef1)
    dbx = np.vstack((dc, dd, de, df, dg, da, db)) # vertical stack
    
    # Datatest: 20 samples/class
    xc = xctest('c',frame1,dstcut1,wave1,coef1)
    xd = xctest('d',frame1,dstcut1,wave1,coef1)
    xe = xctest('e',frame1,dstcut1,wave1,coef1)
    xf = xctest('f',frame1,dstcut1,wave1,coef1)
    xg = xctest('g',frame1,dstcut1,wave1,coef1)
    xa = xctest('a',frame1,dstcut1,wave1,coef1)
    xb = xctest('b',frame1,dstcut1,wave1,coef1)
    testx = np.vstack((xc, xd, xe, xf, xg, xa, xb)) # vertical stack

    # Recognition
    numtest = 140     # number of all test chords
    nsctest = 20      # number of sample in every class of test chords
    chordout1 = ['z']*numtest
    true1 = np.zeros(numtest)
    
    for m in range(numtest):
        classtc = int(np.floor(m/nsctest))      # class of test chord
        chordout1[m],true1[m] = chordclass(dbx,testx[m,:],classtc)
    
    truenumber = sum(true1)
    accuracy = (truenumber/140)*100
    return accuracy

# =============================================================
# Combined dst-dwt feature extraction for guitar chords recognition
def dstdwtwin (dstcut,wave):
    framelist = np.array([128,256,512,1024,2048])
    for m in range(5):
        framelen = framelist[m]
        k = 0;
        accuracy = np.zeros(8)
        for numxc in range(8):
            accuracy[k] = datarecog(framelen,dstcut,wave,numxc+1)
            k = k+1;
        
        np.set_printoptions(precision=2,floatmode='fixed')
        print('FrameLength = ',f'{framelen:4d}',
              ', Accuracy = ',accuracy)
    print('')
    return

# =============================================================
# Main Program
# =============================================================
dstcutlist = np.array([1, 0.5, 0.25])
wavelist = (['db1','db2','db3','db4','db5','db6',
            'bior1.1','bior1.3','bior1.5',
            'bior2.2','bior2.4','bior2.6',
            'bior3.1','bior3.3','bior3.5'])

for k in range(len(dstcutlist)):
    for m in range(len(wavelist)):
        print('DSTCuttingFactor =',dstcutlist[k],
              ', WaveFilter = ',wavelist[m])
        dstdwtwin(dstcutlist[k],wavelist[m])

# ==============================================================


