# %%
import os
import numpy as np
from image_analysis import Image_Analysis
from sound_synthesis import Audio_Synthesis

class Image_to_Audio():
    def __init__(self, fn=None, name="", duration_seconds=30, samplerate=16000):
        self.fn=fn        
        self.img_a=Image_Analysis(fn=fn, name=name)
        self.a_syn=Audio_Synthesis(duration_seconds=duration_seconds, samplerate=samplerate)
        self.sr=self.a_syn.sr
    def insert_background_noise(self, means, stddevs, max_contours):
        for i in range(len(means)):
            mean_c=means[i]
            std_c=stddevs[i]
            ampl_range = np.clip(std_c/128,0,1)
            
            ampl_min = (1-ampl_range)/2
            ampl_max = 1-ampl_min
            period_seconds=np.random.rand()
            period_seconds= (1+period_seconds) * (max_contours /max(self.img_a.im.shape[0],self.img_a.im.shape[1]))
            
            freq = 1/np.clip(period_seconds,1,10)
            
            env = self.a_syn.create_env(type_env="cos", ampl_min=ampl_min, 
                                                        ampl_max=ampl_max, 
                                                        freq=freq)
            fc_low=None
            if mean_c>128: # very high
                fc_high = self.sr/8 * mean_c/64                
            elif mean_c>64:
                fc_high = self.sr/4 * mean_c/128
            else:
                fc_high=self.sr/2 * mean_c/256
                fc_low=.8*fc_high
            self.a_syn.add_noise(ampl=0.2,fc_low=fc_low, fc_high=fc_high,env=env)
        

    def convert(self):
        means, stddevs=self.img_a.get_stats()
        print(means, stddevs)
        contours, len_contours=self.img_a.get_contours()
        max_contours=np.max(len_contours)
        lines=self.img_a.get_lines()
        self.insert_background_noise(means, stddevs, max_contours)
        
    def show(self):
        self.img_a.show()
    def write(self, fn):
        self.a_syn.generate()
        self.a_syn.write(fn)

#%%
if __name__=="__main__":
    IA=Image_to_Audio(fn="../images/milan.jpg", name="beach")
    IA.convert()
    if IA.fn=="../images/beach.jpg":
        fn_out=os.path.join("../audio_test/test0beach.wav")
    if IA.fn=="../images/milan.jpg":
        fn_out=os.path.join("../audio_test/test0milan.wav")
    IA.write(fn_out)
    IA.show()
