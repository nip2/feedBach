import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile as wv
from datetime import date
pi = np.pi


######################
# DSP functions
######################

# setup plotting dimensions
# input: T seconds long, fs is sample rate (Hz)
def dim_def(T,fs):
    N = round(T * fs)     # number of samples
    n = np.linspace(0,N,N)
    Ts = 1/fs        # delta t
    #Ws = 2*np.pi*fs  # sampling freq (rad)
    
    return Ts,N,n

# func to zero pad
# upsample to power of 2 for N-point (Np) FFT
def zeropad(x,N,fs,trunc):
    # with zero padding, make Np at least 2*N
    k = 5          # N to the kth power
    Np = 2**k
    while N > Np:
        k = k + 1
        Np = 2**k

    f = np.linspace(0, fs, Np)
    Ws = 2*np.pi*fs  # sampling freq (rad)

    # zero pad
    C = int((Np-N)/2)       # index at t = 0
    D = C + N               # index at t = T
    
    z = np.zeros(Np)
    z[C:D] = x
    x = z
    
    plotlen = int(round(Np/trunc))
    return x,Np,plotlen,C,D,Ws

# function to plot fft
# trunc => plotlength = Np/trunc
# line True/False if you want to plot vertical line at max amp freq
def plot_fft(x,x_title,N,fs,trunc,line):
    plotlen = int(round(N/trunc))
    
    #x,N,plotlen,C,D,_ = zeropad(x,N,fs,trunc)
    f = np.linspace(0, fs, N)
    
    Y = scipy.fft.fft(x)
    Y_mag = 20*np.log10(np.absolute(Y))
    fmax = f[np.argmax(Y_mag)]
    
    plt.figure(figsize=(14, 5))
    plt.title('FFT of %s' % x_title)
    plt.plot(f[:plotlen], Y_mag[:plotlen])
    
    if line == True:
        plt.vlines(fmax,np.amin(Y_mag[:N//4]),np.amax(Y_mag[:N//4]),colors='r',label='%.1f Hz' %(fmax))
        plt.legend()
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    #plt.show()
    
# get the moving average of an array using a window of length M samples
def movAvg(M,array):
    mov = 1/M * np.ones(M)
    smooth = np.convolve(array,mov,'same')
    return smooth

# fade in/fade out for audio output
# N samples, sig time series
def fade(N,sig):
    fadelen = int(round(1/10 * N))
    output = sig
    for k in range(fadelen):
        output[k] = k/fadelen * output[k]
    for g in range(fadelen):
        q = (N-1) - g
        output[q] = g/fadelen * output[q]
    return output

# create audio file
def write_audio(array,fs,name,sCount):
    today = date.today()
    d8 = today.strftime("%b-%d-%Y")
    stitle = '%s_%d_%s.wav' %(name,sCount,d8)
    
    # normalize audio
    norm = np.linalg.norm(array)
    norm_array = array/norm
    
    sCount = sCount + 1
    wv.write(stitle, fs, norm_array)

# find closest number to K in array
# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(lst, K):
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return lst[idx],idx
    
def getList(dict): 
    keylist = []
    for key in dict.keys():
        keylist.append(key)
    return keylist
    
# Function to find all the local maxima  
# and minima in the given array arr[]
# https://www.geeksforgeeks.org/find-indices-of-all-local-maxima-and-local-minima-in-an-array/
def findLocalMaximaMinima(n, arr):  
    # Empty lists to store points of  
    # local maxima and minima  
    mx = []  
    mn = []  
  
    # Checking whether the first point is  
    # local maxima or minima or neither  
    if(arr[0] > arr[1]):  
        mx.append(0)  
    elif(arr[0] < arr[1]):  
        mn.append(0)  
  
    # Iterating over all points to check  
    # local maxima and local minima  
    for i in range(1, n-1):  
  
        # Condition for local minima  
        if(arr[i-1] > arr[i] < arr[i + 1]):  
            mn.append(i)  
  
        # Condition for local maxima  
        elif(arr[i-1] < arr[i] > arr[i + 1]):  
            mx.append(i)  
  
    # Checking whether the last point is  
    # local maxima or minima or neither  
    if(arr[-1] > arr[-2]):  
        mx.append(n-1)  
    elif(arr[-1] < arr[-2]):  
        mn.append(n-1)
    
    # returns indexes of max (mx) and min (mn) magnitudes in arr
    return mx,mn


######################
# Waveform functions
######################

# function for summing two notes by interval
# N in samples, fs in Hz, fund is string ('A4'),
# interval is an integer (num semitones)
# pitch is given by pitch_dict_create()
def two_pure_tones(N,fs,fund,pitch,interval,notes):
    
    # find fund, select note by given interval:
    f0 = pitch[fund]
    temp = list(pitch)
    try: 
        res = temp[temp.index(fund) + interval] 
    except (ValueError, IndexError): 
        res = None
    fb = pitch[str(res)]
    
    # define dimensions
    Ts = 1/fs
    n = np.linspace(0,N,N)
    w0 = 2*pi*f0
    wb = 2*pi*fb
    
    sin_0 = np.sin(w0*Ts*n)
    sin_b = np.sin(wb*Ts*n)
    x = sin_0 + sin_b
    x_title = '%d half-step interval, %s from %.1f Hz' % (interval,notes['intervals'][interval],f0)
    
    # output is x[] time series, x title string
    return x,x_title

# function for summing odd or even harmonics
# harm is a string ('even','odd','overlay')
# num_harm is an integer
def simple_harmonic(N,fs,fund,pitch,harm,num_harm):
    f0 = pitch[fund]
    
    # define dimensions
    Ts = 1/fs
    n = np.linspace(0,N,N)
    w0 = 2*pi*f0
    
    if harm == 'even':
        sig = np.zeros(N)
        for m in range(num_harm):
            even = 2*(m+1)
            harm = (2/even)*np.sin(w0*even*Ts*n)
            sig = sig + harm
        x = sig
        x_title = 'sum of %d even harmonics' % num_harm
    elif harm == 'odd':
        sig = np.zeros(N)
        for m in range(num_harm):
            odd = 2+(2*m-1)
            harm = (1/odd)*np.sin(w0*odd*Ts*n)
            sig = sig + harm
        x = sig
        x_title = 'sum of %d odd harmonics' % num_harm
    elif harm == 'overlay':
        fig= plt.figure(figsize=(15,6))
        #t = np.linspace(0,Ts*N,N)
        
        Tp = 1/f0
        Nt = int(round(Tp/Ts))
        t = np.linspace(0,Ts*Nt,Nt)
        
        x = np.zeros(N)
        for m in range(num_harm):
            harm = (1/(1+m))*np.sin(w0*(1+m)*Ts*n)
            x = x + harm
            x_title = 'first %d harmonics from %s' % (num_harm,fund)
            plt.plot(t,harm[:len(t)])
            plt.title(x_title)
            plt.xlabel('time (s)')
        plt.show()
    # return chosen time series, title
    return x,x_title

# plot waveform
def plot_waveforms(Ts,N,x,x_title):
    t = np.linspace(0,Ts*N,N)
    fig= plt.figure(figsize=(14,5))

    plt.plot(t,x)
    plt.title(x_title)
    plt.xlabel('time (s)')
    plt.ylabel('magnitude')
    
# plot waveform by frequency
def plot_freq(Ts,N,f1,f2):
    n = np.linspace(0,N,N)
    t = np.linspace(0,Ts*N,N)
    w1 = 2*pi*f1
    w2 = 2*pi*f2
    sin_1 = np.sin(w1*Ts*n)
    sin_2 = np.sin(w2*Ts*n)
    x = sin_1 + sin_2
    
    fig= plt.figure(figsize=(14,5))
    plt.plot(t,x)
    plt.title('%.1f Hz and %.1f Hz' % (f1,f2))
    plt.xlabel('time (s)')
    plt.ylabel('magnitude')
    
    return x


######################
# Musical definitions
######################

# define note names and pitches
def pitch_dict_create():
    
    # init note names and interval names
    note = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    tet = len(note)
    intervals = ['unison','min 2nd','maj 2nd','min 3rd','maj 3rd','prf 4th',
             'tritone','prf 5th','min 6th','maj 6th','dom 7th','maj 7th','octave']
    # init vars
    octave = 0
    count = 0
    note_names = []
    pitch = {}

    # 9 octaves of note names in scientific format
    # ex. A4 is A in the 4th octave, middle A
    for number in range(108):
        note_names.append("%s%d" %(note[number % 12], octave))
        count = count+1
        if count == 12:
            octave = octave+1
            count = 0

    # 9 octaves of pitches matched with note names
    A4 = 440     # A440 pitch standard 440 Hz
    for k in range(108):
        pitch[note_names[k]] = A4 * (2**(1/12))**(k-57)

    # above pitch equation:
    # all notes relative to standard A440 pitch
    # frequency of a pitch relative to standard is:
    # ratio = (2^(1/12))^n
    # where n is the # of half-steps from the standard to the pitch
    # n = k-57 such that n=0 corresponds to A440 with 108 pitches
    
    notes = {'intervals':intervals,'names':note_names,'pitch':pitch}
    return notes
    
# need functions to determine how close 
# the given interval is to a 'simple' one
# the closer to a simple ratio, the more
# consonant the interval

# get ratio info from notes
# b is the higher note, in Hz
def f_to_ratio(b,a):
    b = int(round(b))
    a = int(round(a))
    div = np.gcd(b,a)
    nume = int(b/div)
    deno = int(a/div)
    rat = nume/deno
    return nume,deno,div,rat

# get closest simple ratio
# from num/denom given by f_to_ratio()
def close_to_simple_ratio(nume,deno):
    pmin_diff = 1000        # set absurdly high
    for i in range(12):
        for j in range(11):
            diff = np.abs(nume/deno - i/(j+1))
            if diff < pmin_diff:
                pmin_diff = diff
                closest_num = i
                closest_den = (j+1)
            else:
                diff = diff
    return closest_num,closest_den,pmin_diff

# function that takes a note and interval, returns the note at interval
# fund is string name, interval is int
def note_interval(fund,interval,note_names):
    fund_index = note_names.index(fund)
    note_index = fund_index + interval
    note = note_names[note_index]
    
    return note

# lists a given scale by name from scales dictionary
# for a given root note
def list_scale(root,name,scales,note_names):
    scale = []
    root_index = note_names.index(root)
    steps = scales[name]
    for k in range(len(steps)):
        index = root_index + sum(steps[:k])
        scale.append(note_names[index])
        
    return scale
