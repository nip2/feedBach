#%%
from functions import *
from fractions import Fraction

#%%
# make lists of various scale intervals
# scales are listed by step sizes in semitones
scales = {
    'major':      [2,2,1,2,2,2,1],
    'dimW':       [2,1,2,1,2,1,2,1],
    'dimH':       [1,2,1,2,1,2,1,2],
    'pentmaj':    [2,2,3,2,3],
    'bebopmaj':   [2,2,1,2,1,1,2,1],
    'harmmaj':    [2,2,1,2,1,3,1],
    'lydaug':     [2,2,2,2,1,2,1],
    'augment':    [3,1,3,1,3,1],
    'blues':      [3,2,1,1,3,2]
}

# generate modes from major
modes = ['ionian','dorian','phrygian','lydian',
         'mixolydian','aeolian','locrian']
for k in range(6):
    scales[modes[k+1]] = np.roll(scales['major'],-(k+1))

# init vars
sCount = 0

#%%
# setup note elements
notes = pitch_dict_create()
intervals = notes['intervals']
note_names = notes['names']
pitch = notes['pitch']

# setup dimensions
T = 1/27.5      # T seconds long
fs = 22050      # Hz
nyq = round(int(fs/2))  # highest freq before aliasing
Ts,N,n = dim_def(T,fs)

# create and plot odd/even harmonic sums
x,x_title = simple_harmonic(N,fs,'A2',pitch,'odd',11)
plot_waveforms(Ts,N,x,x_title)

x,x_title = simple_harmonic(N,fs,'A2',pitch,'even',11)
plot_waveforms(Ts,N,x,x_title)

# plot an overlay of the harmonics
# A0 is at 27.5 Hz, N = 800, therefore:
# 1/(f_A0) = N/fs length of plot matches period of pitch

# time series plot over longer duration
Ts,N,n = dim_def(.5,fs)
x,x_title = simple_harmonic(N,fs,'A0',pitch,'overlay',12)
xt_title = x_title + ' time series'
plot_waveforms(Ts,N,x,xt_title)

# FFT plot
plot_fft(x,x_title,N,fs,20,True)

#%%
# interval analysis
# use a specific N value:
N = 1200
sCount = 0

# plot 12 intervals
offset = note_names.index('A4')  # starting index at fund
interval_ratio = {}
f0 = pitch['A4']     # fund
for j in range(13):
    
    # get interval stats
    fb = pitch[note_names[offset+j]]
    nume,deno,div,rat = f_to_ratio(fb,f0)
    c_num,c_den,diff = close_to_simple_ratio(nume,deno)
    interval_ratio[intervals[j]] = Fraction(c_num,c_den)
    
    # print interval stats
    print('interval: %s, %s' % (intervals[j],note_names[offset+j]))
    print('given note ratio: %d / %d = %.5f with gcd %d' %(nume,deno,rat,div))
    print('closest simple ratio: %d / %d' %(c_num,c_den))
    print('absolute diff in ratio btwn just and equal temperament: %.5f' % diff)
    
    # plot sum of interval
    x,x_title = two_pure_tones(N,fs,'A4',pitch,j,notes)
    plot_waveforms(Ts,N,x,x_title)
    plt.show()

#%%
