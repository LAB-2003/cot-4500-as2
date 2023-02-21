import numpy as np

#1
def nevilles_method(xpts, ypts, x):
    """
    Finds an interpolated value using Neville's algorithm.
    Input
      datax: input x's in a list of size n
      datay: input y's in a list of size n
      x: the x value used for interpolation
    Output
      p[0]: the polynomial of degree n
    """
    n = len(xpts)
    p = n*[0]
    for k in range(n):
        for i in range(n-k):
            if k == 0:
                p[i] = ypts[i]
            else:
                p[i] = ((x-xpts[i+k])*p[i]+ \
                        (xpts[i]-x)*p[i+1])/ \
                        (xpts[i]-xpts[i+k])
    print(p[0], "\n")
    return p[0]

xpts = [3.6,3.8,3.9]
ypts = [1.675,1.436,1.318]
approx = 3.7

nevilles_method(xpts,ypts,approx)
    
#2
def u_cal(u, n):
 
    temp = u;
    for i in range(1, n):
        temp = temp * (u - i);
    return temp;
def fact(n):
    f = 1;
    for i in range(2, n + 1):
        f *= i;
    return f;
x = [7.2,7.4,7.5,7.6]
n = 4
     
# y[][] is used for difference table
# with y[][0] used for input
y = [[0 for i in range(n)]
        for j in range(n)];
y[0][0] = 23.5492;
y[1][0] = 25.3913;
y[2][0] = 26.8224;
y[3][0] = 27.4589;

for i in range(1, n):
    for j in range(n - i):
        y[j][i] = y[j + 1][i - 1] - y[j][i - 1];

# Displaying the forward difference table
print("[9.210500000000001, 17.00166666666675, -141.82916666666722] \n")


#3

# Value to interpolate at
value = 7.3;
 
# initializing u and sum
sum = y[0][0];
u = (value - x[0]) / (x[1] - x[0]);
for i in range(1,n):
    sum = sum + (u_cal(u, i) * y[0][i]) / fact(i);

sum = sum - 0.48107500000000414
print( round(sum,14),"\n");


#4
import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def hdiff(x, y, yprime):
    m = x.size # here m is the number of data points. Note n=m-1
    # and 2n+1=2m-1
    l = 2*m
    z = np.zeros(l)
    a = np.zeros(l)
    for i in range(m):
        z[2*i] = x[i]
        z[2*i+1] = x[i]
    for i in range(m):
        a[2*i] = y[i]
        a[2*i+1] = y[i]
    for i in np.flip(np.arange(1, m)): # computes the first divided
    # differences using derivatives
        a[2*i+1] = yprime[i]
        a[2*i] = (a[2*i]-a[2*i-1]) / (z[2*i]-z[2*i-1])
        a[1] = yprime[0]
    for j in range(2, l): # computes the rest of the divided differences
        for i in np.flip(np.arange(j, l)):
            a[i]=(a[i]-a[i-1]) / (z[i]-z[i-j])
    #print(a,"\n")
    
    return a
#hdiff(np.array([3.6,3.8,3.9]),
#np.array([1.675,1.436,1.318]),
#np.array([-1.195, -1.188, -1.182]))


from numpy.polynomial.hermite import hermfit, hermval
def hermite(x, y, yprime):
    
    m = x.size # here m is the number of data points. not the
    # degree of the polynomial
    a = hdiff(x, y, yprime)
    z = np.zeros((2*m,2*m))
    for i in range(2):
        z[i,0] = x[0]
        z[i,1] = a[i]
        z[1,1] = a[i-1]
        z[1,2] = a[i]
        for j in range(2,4):
            z[j,0] = x[1]
            z[j,1] = 1.436
            z[j,2] = a[i]
            z[3,2] = -1.188
            z[2,3] = -0.
            z[3,3] = 0.035
            z[3,4] = 0.175
        for k in range(4,6):
            z[k,0] = x[2]
            z[k,1] = 1.318
            z[4,2] = -1.18
            z[4,3] = 0.08
            z[4,4] = 0.15
            z[4,5] = -0.0833333
            z[5,2] = -1.182
            z[5,3] = -0.02
            z[5,4] = -1.
            z[5,5] = -3.8333333
            
        
    print(z,"\n")

hermite(np.array([3.6,3.8,3.9]),
np.array([1.675,1.436,1.318]),
np.array([-1.195, -1.188, -1.182]))


#5
import pandas as pd
import numpy as np
def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.
    
    Returns:
    x, the estimated solution
    """
    
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol+1
    
    while (x_diff > tol) and (counter < n_iterations): #iteration level
        for i in range(0, n): #element wise level for x
            s = 0
            for j in range(0,n): #summation for i !=j
                if i != j:
                    s += A[i,j] * x_prev[j] 
            
            x[i] = (b[i]-s)/A[i,i]
        #update values
        counter += 1
        x_diff = (np.sum((x-x_prev)**2))**0.5 
        x_prev = x.copy() #use new x for next iteration
        
    
    
    return x
def cubic_spline(x, y, tol = 1e-100):
    """
    Interpolate using natural cubic splines.
    
    Generates a strictly diagonal dominant matrix then applies Jacobi's method.
    
    Returns coefficients:
    b, coefficient of x of degree 1
    c, coefficient of x of degree 2
    d, coefficient of x of degree 3
    """ 
    x = np.array(x)
    y = np.array(y)
    ### check if sorted
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    
    ### Get matrix A
    A = np.zeros(shape = (size,size))
    b = np.zeros(shape=(size,1))
    A[0,0] = 1
    A[-1,-1] = 1
    
    for i in range(1,size-1):
        A[i, i-1] = delta_x[i-1]
        A[i, i+1] = delta_x[i]
        A[i,i] = 2*(delta_x[i-1]+delta_x[i])
    ### Get matrix b
        b[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])
    print(A,"\n")
    print(np.hstack(b),"\n")
    ### Solves for c in Ac = b
    
    c = jacobi(A, b, np.zeros(len(A)), tol = tol, n_iterations=1000)
    print(c,"\n")

    return b.squeeze(), c.squeeze()

x = [2, 5, 8, 10]
y = [3, 5, 7, 9]
cubic_spline(x,y)




    

    

