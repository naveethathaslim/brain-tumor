import numpy as np
import matplotlib.pyplot as plt
fs=500
t=np.linspace(0,1,fs)
freq=5
x=np.sin(2*np.pi*freq*t)
plt.figure(figure=(7,7))
plt.plot(t,x)
plt.title("sine wave")
plt.xlabel("Time(s)")
plt.ylabel("amplitude")
plt.show()