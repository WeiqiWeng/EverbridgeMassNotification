import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

w10, w20 = 6.40431588e-02, -3.86624017e-02
w11, w21 = 1.70148834e-02, -3.25246228e-02
w12, w22 = -8.10580422e-02, 7.11870244e-02

x = np.linspace(-10, 10, 1000)

paths_colors = [(197/255, 31/255, 31/255),
                (222/255, 156/255, 83/255),
                (250/255, 218/255, 141/255),
                (160/255, 191/255, 124/255),
                (28/255, 120/255, 135/255),
                (36/255, 169/255, 225/255)]

labels = ['$-w_1 e^{-w_1 x_1}$ -- class 1',
          '$-w_2 e^{-w_2 x_2}$ -- class 1',
          '$-w_1 e^{-w_1 x_1}$ -- class 2',
          '$-w_2 e^{-w_2 x_2}$ -- class 2',
          '$-w_1 e^{-w_1 x_1}$ -- class 0',
          '$-w_2 e^{-w_2 x_2}$ -- class 0']
plt.plot(x, -w10*np.exp(-w10*x), linewidth=2, color=paths_colors[0])
plt.plot(x, -w20*np.exp(-w20*x), linewidth=2, color=paths_colors[1])
plt.plot(x, -w11*np.exp(-w11*x), linewidth=2, color=paths_colors[2])
plt.plot(x, -w21*np.exp(-w21*x), linewidth=2, color=paths_colors[3])
plt.plot(x, -w12*np.exp(-w12*x), linewidth=2, color=paths_colors[4])
plt.plot(x, -w22*np.exp(-w22*x), linewidth=2, color=paths_colors[5])
plt.legend(labels, loc='best')
plt.xlabel('$x_i$')
plt.ylabel('$-w_i e^{-w_i x_i}$')
plt.savefig('../pics/exp_plot.png')
plt.close()

