# %% 
import numpy as np
import soundfile as sf
import scipy
import os
from scipy.signal import lfilter, iirdesign

os.chdir(os.path.abspath(os.path.dirname(__file__)))
freqs_notes=[220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]
# %%

def map_(x, range_in, range_out):
    y=x-range_in[0]
    y/=(range_in[1]-range_in[0])
    y*=(range_out[1]-range_out[0])
    return y+range_out[0]
class Audio_Synthesis():
    def __init__(self, duration_seconds=30, samplerate=16000):
        assert duration_seconds>0, "negative duration!"
        assert samplerate>8000, "samplerate must be at least 8000 Hz"
          
        self.dur=duration_seconds
        self.sr=samplerate
        self.dur_samples=int(self.dur*self.sr)
        self.features=[]
        self.signal=np.zeros((self.dur_samples,))

    def generate(self ):
        if len(self.features)==0:
            print("Sorry, you should give me some features")
            return
        for feature in self.features:
            func=feature["func"]
            func(**feature["kwargs"])
    def create_env(self, type_env="cos", ampl_min=0, ampl_max=1, freq=0, 
                         dur_onset=0.5, onsets = []):
        if type_env=="cos":
            env=np.cos(2*np.pi*freq*np.arange(self.dur_samples)/self.sr)
            env=map_(env, (-1,1), (ampl_min, ampl_max))
        elif type_env=="square" or type_env=="saw":
            event=np.zeros(int(self.sr/freq))
            if type_env=="square":                
                sq4=int(event.size/4)
                event[sq4:3*sq4]=1            
            elif type_env =="saw":
                event=np.linspace(1,0, event.size)
            
            env = np.tile(event, (int(np.ceil(self.dur_samples/event.size)),))
            env = env[:self.dur_samples]
            env = map_(env, (0,1), (ampl_min, ampl_max))

        elif type_env=="onset":
            event=np.sin(np.linspace(0, np.pi, int(dur_onset*self.sr)))
            half_dur=int(dur_onset/2*self.sr)
            env=np.zeros((self.dur_samples))
            for ev in onsets:
                i=int(ev*self.sr)
                i_l = max(0, i-half_dur)
                i_r= min(i+half_dur, self.dur_samples-1)
                env[i_l:i_r] = event[half_dur-i+i_l:half_dur+(i_r-i)]
        return env
    def fix_poles(self, a):
        poles=np.roots(a)
        THR = 0.98
        if not np.any(np.abs(poles)>THR):
            return a
        idxs=np.abs(poles)>THR
        poles[idxs]= THR*poles[idxs]/np.abs(poles[idxs])

        return np.poly(poles)

    def band_pass(self, x, fc_low, fc_high):
        b, a = iirdesign(wp=(fc_low,fc_high),
                         ws=(fc_low-50,fc_high+50),
                         gpass=1, gstop=60,            
                         fs=self.sr)
        a=self.fix_poles(a)
        return lfilter(b,a,x)

    def high_pass(self, x, fc):
        b, a = iirdesign(wp=fc,
                         ws=fc-50,
                         gpass=1, gstop=60,            
                         fs=self.sr)
        a=self.fix_poles(a)
        return lfilter(b,a,x)
        
    def low_pass(self, x, fc):
        b, a = iirdesign(wp=fc,
                         ws=fc+50,
                         gpass=1, gstop=60,            
                         fs=self.sr)
        a=self.fix_poles(a)
        return lfilter(b,a,x)
    def apply_envelope(self, x, env):
        if env is None:
            return x
        return x*env        
    def apply_noise(self, ampl=1, fc_high=None, fc_low=None, env=None):        
        noise = np.random.randn((self.dur_samples))
        noise/= np.sqrt(np.mean(np.power(np.abs(noise),2)))
        noise = noise * ampl
        if fc_low and fc_high is not None:
            noise=self.band_pass(noise, fc_low, fc_high)
        elif fc_low is not None:
            noise=self.high_pass(noise, fc_low)
        elif fc_high is not None:
            noise=self.low_pass(noise, fc_high)
        noise=self.apply_envelope(noise, env)
        
        self.signal+=noise
    def add_noise(self, ampl=1, fc_high=None, fc_low=None, env=None):        
        self.features.append({"func":self.apply_noise,
                              "kwargs":{"ampl":ampl, "fc_high":fc_high, 
                                        "fc_low":fc_low, "env":env}
                            })
    def __add_(self, what, **kwargs):
        self.features.append({"func":what, "kwargs":kwargs})
    def apply_saw(self, ampl=1,freq=0, env=None):
        saw = self.create_env(type_env="saw", ampl_min=-1, ampl_max=1, freq=freq)
        saw=self.apply_envelope(saw, env)        
        self.signal+=saw
    def apply_tone(self, ampl=1,freq=0, env=None):
        tone = self.create_env(type_env="cos", ampl_min=-1, ampl_max=1, freq=freq)
        tone=self.apply_envelope(tone, env)        
        self.signal+=tone
    
    def apply_square(self, ampl=1,freq=0, env=None):
        square = self.create_env(type_env="square", ampl_min=-1, ampl_max=1, freq=freq)
        square = self.apply_envelope(square, env)        
        self.signal+=square

    def add_square(self, ampl=1, freq=0, env=None):        
        self.__add_(self.apply_square, ampl=ampl, freq=freq, env=env)
    def add_tone(self, ampl=1, freq=0, env=None):        
        self.__add_(self.apply_tone, ampl=ampl, freq=freq, env=env)
    def add_saw(self, ampl=1, freq=0, env=None):        
        self.__add_(self.apply_saw, ampl=ampl, freq=freq, env=env)
    
        
    def normalize(self):
        if np.all(self.signal==0):
            return
        self.signal=.707*self.signal/np.max(np.abs(self.signal))
        
    def write(self, fn="tmp.wav"):
        self.normalize()
        if not fn.endswith(".wav"):
            raise NotImplementedError("sorry, only wav files are currently supported")
        sf.write(fn, self.signal, self.sr)
        
# %% Test

if __name__=="__main__":
    as0=Audio_Synthesis(30, 16000)
    as0.generate()
    as0.write("../audio_test/test0.wav")

    
    as1=Audio_Synthesis(30, 16000)
    env = as1.create_env(ampl_min=0.2, ampl_max=1, freq=1/3)
    as1.add_noise(fc_high=500, env= env)
    as1.generate()
    as1.write("../audio_test/test1.wav")

    as2=Audio_Synthesis(30, 16000)
    as2.add_noise(fc_high=500, fc_low=440, env=env)
    as2.generate()
    as2.write("../audio_test/test2.wav")

    as3=Audio_Synthesis(30, 16000)
    env = as3.create_env(ampl_min=0.2, ampl_max=1, freq=1/5)
    as3.add_noise(fc_high=1000, fc_low=100, env=env)
    as3.generate()
    as3.write("../audio_test/test3.wav")


    as4=Audio_Synthesis(30, 16000)
    onset_events=np.random.rand(np.random.randint(5,15))*30
    env4 = as4.create_env(type_env="onset", onsets=onset_events)
    as4.add_noise(fc_high=500, fc_low=400, env=env4)
    as4.generate()
    as4.write("../audio_test/test4.wav")

    as5=Audio_Synthesis(30, 16000)
    n_bands=5
    LOW_BAND=100
    HIGH_BAND=0.9*as5.sr/2
    size_band=np.log(HIGH_BAND-LOW_BAND)/n_bands
    amplitudes=np.log10(np.logspace(1,0.2, n_bands+1))
    for i in range(-1, n_bands):
        
        ampl_max=0.1+(1-0.1-amplitudes[i+1])*np.random.rand()
        ampl_min=np.random.rand()*ampl_max*.8
        env = as5.create_env(ampl_min=ampl_min, ampl_max=ampl_max, freq=1/np.random.randint(1,10))
        if i>=0:
            freq_left=LOW_BAND+np.exp(i*size_band) 
            freq_right=LOW_BAND+np.exp((i+1)*size_band) 
        else:
            freq_right=LOW_BAND
            freq_left=None
        as5.add_noise(fc_high=freq_right, fc_low=freq_left, env=env.copy())

    as5.generate()
    as5.write("../audio_test/test5.wav")


    as6=Audio_Synthesis(30, 16000)
    

    for _ in range(5): # type of notes
        onset_events=np.random.rand(np.random.randint(5,15))*30
        env6 = as6.create_env(type_env="onset", onsets=onset_events, 
                              dur_onset = .1+np.random.rand()*.4)

        as6.add_saw(freq=freqs_notes[np.random.randint(len(freqs_notes))] , 
                     ampl=np.random.rand()*.8+.1, env=env6.copy())
    as6.generate()
    as6.write("../audio_test/test6.wav")


# %%
