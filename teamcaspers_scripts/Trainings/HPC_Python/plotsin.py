# coding: utf-8
get_ipython().magic('pylab inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
x = linspace(0, 2 * np.pi, 120)
y = sin(x)
plot(x, y, label=r"$\sin(x)$")
xlim([0, 2 * np.pi])
grid(1)
xticks(np.linspace(0, 2 * np.pi, 9), [0, r"$\frac{\pi}{4}$", 
                                          r"$\frac{\pi}{4}$", r"$\frac{3\pi}{4}$", r"$\pi$", 
                                          r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"
                                          , r"$2\pi$"])
xlabel('x')
legend()
