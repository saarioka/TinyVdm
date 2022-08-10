import scipy
import numpy as np
import matplotlib.pyplot as plt

### The q-Gaussian distribution

### From Appendix A of https://arxiv.org/pdf/1806.07317.pdf

def C_q(q):
    Gamma=scipy.special.gamma
    sqrt=np.sqrt
    pi=np.pi
    if q<1:
        return (2*sqrt(pi))/((3-q)*sqrt(1-q))*(Gamma((1)/(1-q)))/(Gamma((3-q)/(2*(1-q))))
    elif q==1:
        return sqrt(pi)
    elif q<3:
        return (sqrt(pi))/(sqrt(q-1))*(Gamma((3-q)/(2*(q-1))))/(Gamma((1)/(q-1)))
    else:
        raise Exception('Please q<3!!!')

def e_q(x,q):
    if q==1:
        return np.exp(x)
    elif (1+(1-q)*x)>0:
        return (1+(1-q)*x)**(1/(1-q))
    else:
        return 0

def qgauss(x, q, b):
    return np.sqrt(b)/C_q(q)*e_q(-b*x**2,q)

qgauss_v = np.vectorize(qgauss)

x=np.linspace(-5,5,1001)

plt.plot(x,qgauss_v(x,0,1), 'b', lw=2.5, label='q=0, b=1')
plt.plot(x,qgauss_v(x,1,1), 'k', lw=2.5, label='q=1, b=1')
plt.plot(x,qgauss_v(x,2,1), 'r', lw=2.5, label='q=2, b=1')
plt.plot(x,qgauss_v(x,2,2), 'g', lw=2.5, label='q=2, b=2')
plt.ylim(0.,.8)
plt.legend(loc='best')
plt.grid()
