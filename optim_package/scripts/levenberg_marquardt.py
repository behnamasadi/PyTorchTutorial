#Minimizing a function is like trying to find lowest point on a surface. Think of yourself walking on a hilly surface and that you are trying to get to the lowest point. You would find the direction that goes downhill and walk until it doesn't go downhill anymore. Then you would chose a new direction that goes downhill and walk in that direction until it doesn't go downhill anymore, and so on. Eventually (hopefully) you would reach a point where no direction goes downhill anymore. You would then be at a (local) minimum.

#The LM algorithm, and many other minimization algorithms, use this scheme.

#Suppose that the function being minimized is F and we are at the point x(n) in our iteration. We wish to find the next iterate x(n+1) such that F(x(n+1)) < F(x(n)), i.e. the function value is smaller. In order to chose x(n+1) we need two things, a direction from x(n) and a step size (how far to go in that direction). The LM algorithm determines these values as follows -

#First, compute a linear approximation to F at the point x(n). It is easy to find out the downhill direction of a linear function, so we use the linear approximating function to determine the downhill direction. Next, we need to know how far we can go in this chosen direction. If our approximating linear function is a good approximation for F for a large area around x(n), then we can take a fairly large step. If it's a good approximation only very close to x(n), then we can take only a very small step.

#This is what LM does - calculates a linear approximation to F at x(n), thus giving the downhill direction, then it figures out how big a step to take based on how well the linear function approximates F at x(n). LM figures out how good the approximating function is by basically taking a step in the direction thus determined and comparing how much the linear approximation to F decreased to the how much the the actual function F decreased. If they are close, the approximating function is good and we can take a little larger step. If they are not close then the approximation function is not good and we should back off and take a smaller step.

#∇
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
#import numpy as np

#import pylab as pl


#fig = plt.figure()
#ax = fig.gca(projection='3d')
#x_start=-4
#y_start=-4

#x_end=4
#y_end=4

#x_step=0.1
#y_step=0.1


#X=np.arange(x_start,x_end,x_step)
#Y=np.arange(y_start,y_end,y_step)

#X, Y = np.meshgrid(X, Y)

#Z=X**2 +((Y**2)/4 +4*(X-Y)**2)*(np.sin(Y)**2)

#plot_surface
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()




#CS = pl.contour(X, Y, Z)
#CB = pl.colorbar(CS, shrink=0.8, extend='both')
#pl.scatter(X, Y, .8)

#pl.axis('tight')
#pl.show()


#https://people.duke.edu/~ccc14/sta-663/CalibratingODEs.html
#https://gist.github.com/geggo/92c77159a9b8db5aae73


#!/usr/bin/python
#-*- coding: latin-1 -*-
"""This module contains pure Python implementations of the
Levenberg-Marquardt algorithm for data fitting.
"""

import numpy

from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve


import time

def gauss1d(pars, x, v0 = 0):
    """calculate 1d gaussian.
    @return: difference of 1d gaussian and reference (data) values
    @param pars: parameters of gaussian. see source.
    @param x: x values
    @param v0: reference value
    """
    A, m, s, offs = pars[0:4]
    v = A*numpy.exp(- (x-m)**2 / (2*s**2)) + offs
    return v-v0

def Dgauss1d(pars, x, v=0):
    """
    calculated Jacobian matrix for 1d gauss
    """
    A, m, s, offs = pars[0:4]
    f = A*numpy.exp( - (x-m)**2 / (2*s**2))
    J = numpy.empty(shape = (4,)+x.shape, dtype = numpy.float_)
    J[0] = 1.0/A * f
    J[1] = f*(x-m)/s**2
    J[2] = f*(x-m)**2/s**3
    J[3] = 1
    return J

def fJ(pars, x, y = 0):
    "Calculation of function and Jacobian for one-dimensional Gaussian."
    A, m, s, offs = pars[0:4]
    f = A*numpy.exp( - (x-m)**2 / (2*s**2))
    if 1:
        J = numpy.empty(shape = (4,)+x.shape, dtype = numpy.float_)
        J[0] = 1.0/A * f
        J[1] = f*(x-m)/s**2
        J[2] = f*(x-m)**2/s**3
        J[3] = 1
        return f + (offs - y), J
    return f + (offs - y)


def rF(pars, x):
    """calculate all f_i and df_i/dp_j"""
    m, s = pars

    #F: function in parts which are then linearly combined to yield total function
    F = numpy.empty(shape = (2,) + x.shape)
    F[0] = numpy.exp( - (x-m)**2 / (2*s**2))
    F[1] = 1

    Fd = numpy.empty(shape = (2,) + F.shape)
    ##Ableitungen nach nichtlinearen Parametern
    #Fd[0]: Ableitungen der F[i] nach m
    Fd[0][0] = F[0] * (x-m)/(s**2)
    Fd[0][1] = 0

    Fd[1][0] = F[0] * (x-m)**2/(s**3)
    Fd[1][1] = 0

    return F, Fd

def fJr(pars, x, y = 0, calcJ = True):
    """
    calculate f and J for reduced system (only nonlinear parameters)
    """

    F, Fd = rF(pars, x)

    #calculate linear Parameters
    FtF = inner(F, F)
    Fty = inner(F, y)
    c = solve(FtF, Fty)

    #calculate residuum
    r = dot(c, F) - y

    if not calcJ:
        return r, c, F

    ##calculate complete Jacobian
    cd = numpy.empty(shape = (len(pars),) + c.shape)
    Jr = numpy.empty(shape = (len(pars),) + x.shape)
    for j in range(len(pars)):
        cd[j] = solve(FtF, inner(Fd[j], r) - inner(F, dot(c, Fd[j])))
        Jr[j] = dot(c, Fd[j]) + dot(cd[j], F)

    return r, Jr

def fitparerror(fitpar, J, res):
    """
    fitpar: fit parameters
    J: Jacobi matrix
    r: residuum
    """
    import scipy.stats
  
    alpha = 0.05 #confidence level 95% for alpha = 0.05, 2sigma confidence limit
    
    N = max(J.shape) #number of points
    m = len(fitpar) #number of parameters

    rnorm = sum(res*res) #norm residuum
  
    sigma = numpy.sqrt(rnorm/(N - m)) # estimated standard deviation
    
    #R = numpy.linalg.qr(numpy.transpose(J), mode='r') # %QR Zerlegung
    #Rinv = numpy.linalg.inv(R) #R\eye(size(R)); 
    #diagonale = numpy.sum((Rinv*Rinv),1) #diagonal entries of inv(R^T R)

    diagonale = numpy.diagonal( numpy.linalg.inv(numpy.inner(J,J)))

    parerr = numpy.sqrt(diagonale) * sigma * scipy.stats.t.ppf(1-alpha/2, N-m);
    parerrrel = parerr/fitpar #relative error

    return parerr, sigma

    
def LM(fun, pars, args,
       tau = 1e-2, eps1 = 1e-6, eps2 = 1e-6, kmax = 20,
       verbose = False,
       full_output = False):
    """Implementation of the Levenberg-Marquardt algorithm in pure
    Python. Solves the normal equations."""
    p = pars
    f, J = fun(p, *args)

    A = inner(J,J)
    g = inner(J,f)

    I = eye(len(p))

    k = 0; nu = 2
    mu = tau * max(diag(A))
    stop = norm(g, Inf) < eps1
    while not stop and k < kmax:
        k += 1

        try:
            d = solve( A + mu*I, -g)
        except numpy.linalg.LinAlgError:
            print ("Singular matrix encountered in LM")
            stop = True
            reason = 'singular matrix'
            break

        if norm(d) < eps2*(norm(p) + eps2):
            stop = True
            reason = 'small step'
            break

        pnew = p + d

        fnew, Jnew = fun(pnew, *args)
        #rho = (norm(f) - norm(fnew))/inner(d, mu*d - g)  # /2????
        rho = (norm(f)**2 - norm(fnew)**2)/inner(d, mu*d - g)
        
        if rho > 0:
            p = pnew
            A = inner(Jnew, Jnew)
            g = inner(Jnew, fnew)
            f = fnew
            J = Jnew
            if (norm(g, Inf) < eps1): # or norm(fnew) < eps3):
                stop = True
                reason = "small gradient"
                break
            mu = mu * max([1.0/3, 1.0 - (2*rho - 1)**3])
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2*nu

        if verbose:
            print ("step %2d: |f|: %9.6g mu: %8.3g rho: %8.3g"%(k, norm(f), mu, rho))

    else:
        reason = "max iter reached"

    if verbose:
        print (reason)
    
    if not full_output:
        return p
    else:
        return p, J, f #fitparerror(p, J, f)

def LMqr(fun, pars, args,
         tau = 1e-3, eps1 = 1e-8, eps2 = 1e-8, kmax = 100,
         verbose = False):

    from scipy.linalg import lstsq
    import scipy.linalg

    """Implementation of the Levenberg-Marquardt algorithm in pure
    Python. Instead of using the normal equations this version uses QR
    factorization for enhanced accuracy. Significantly slower (factor
    2)."""
    p = pars
    f, J = fun(p, *args)

    A = inner(J,J)
    g = inner(J,f)

    I = eye(len(p))

    k = 0; nu = 2
    mu = tau * max(diag(A))
    stop = norm(g, Inf) < eps1

    while not stop and k < kmax:
        k += 1

        if verbose:
            print ("step %d: |f|: %9.3g mu: %g"%(k, norm(f), mu))

        tic = time.time()
        A = inner(J, J)
        g = inner(J, f)

        d = solve( A + mu*I, -g)
        print ('XX', d, time.time() - tic)

        
        des = numpy.hstack((-f, numpy.zeros((len(p),))))
        Des = numpy.vstack((numpy.transpose(J),
                            numpy.sqrt(mu)*I))

        tic = time.time()
        d0, resids, rank, s = lstsq(Des, des)
        print ('d0', d0, time.time() - tic)

        
        tic = time.time()
        #q, r = scipy.linalg.qr(Des, econ = True, mode = 'qr')
        #d4   = solve(r, inner(numpy.transpose(q), des))
        r = scipy.linalg.qr(Des, econ = True, mode = 'r')
        d4   = scipy.linalg.cho_solve( (r, False), -inner(J, f))
        print ('d4', d4, time.time() - tic)

        
        

        tic = time.time()
        q, r = scipy.linalg.qr(numpy.transpose(J), econ = True, mode = 'qr')
        d3 = solve( r + mu*numpy.linalg.inv(r.transpose()), -inner(numpy.transpose(q),f))
        #d3 = scipy.linalg.cho_solve( (r + mu*numpy.linalg.inv(r.transpose()), False),
        #                             -inner(numpy.transpose(q),f))
        print ('d3', d3, time.time() - tic)

        print (d - d0)
        print (d3 - d0)
        print (d4 - d0)


        if norm(d) < eps2*(norm(p) + eps2):
            stop = True
            reason = 'small step'
            break

        pnew = p + d

        fnew, Jnew = fun(pnew, *args)
        rho = (norm(f) - norm(fnew))/inner(d, mu*d - g) # /2????

        if rho > 0:
            p = pnew
            #A = inner(Jnew, Jnew)
            #g = inner(Jnew, fnew)
            f = fnew
            J = Jnew
            if (norm(g, Inf) < eps1): # or norm(fnew) < eps3):
                stop = True
                reason = "small gradient"
                break
            mu = mu * max(1.0/3, 1 - (2*rho - 1)**3)
            nu = 2
        else:
            mu = mu * nu
            nu = 2*nu

    else:
        reason = "max iter reached"

    if verbose:
        print (reason)
    return p


def testLM():
    #       A   m   s  offs
    pars = [1, 0.1, 1, 0.5]
    
    x = numpy.linspace(-3,3,1001)
    y = gauss1d(pars, x)

    #y+= numpy.random.randn(len(x))
    #            A   m    s  offs
    startpars = [1*2, 0.5, 2, 0.5]

    return LM(fJ, startpars, (x, y), verbose = True, tau = 1e-4)

def testfLMs():
    x = numpy.linspace(-3,3,1001)
    #       A   m   s  offs
    pars = [1, 0.1, 1, 0.5]
    y = gauss1d(pars, x)

    #           m     s
    startpar = [0.5, 2]

    #f,c,F = fJl(startpar, x, y, calcJ = False)
    #f,J   = fJl(startpar, x, y, calcJ = True)

    #print "c: ", c
    #print "f: ", f
    #print "F: ", F
    #print "J: ", J

    p = LM(fJr, startpar, (x, y), verbose = True, tau = 1e-4)

    print (p)

def testLMqr():
    pars = [1, 0.1, 1, 0.5]
    x = numpy.linspace(-5,5,1000001)
    y = gauss1d(pars, x) # + numpy.random.randn(len(x))

    pars2 = [1.1, 0.15, 1.3, 0.2]

    return LMqr(fJ, pars2, (x, y), verbose = True)



if __name__ == '__main__':
    print (testLM())
    #print '-'*40
    #print testLMqr()

    #print (testfLMs())
