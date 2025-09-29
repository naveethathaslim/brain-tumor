import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
sample_rate,data=wavfile.read("sample.wav")
duration=len(data)/sample_rate
time=np.linspace(0,duration,len(data))
plt.figure(figure=(6,6))
plt.plot(time,data)
plt.title("sample Audio signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.show()