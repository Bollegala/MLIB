"""
This module implements the BFGS algorithm for unconstrained optimization.

Danushka Bollegala.
2012/01/29.
"""

import numpy

class BFGS:

    def __init__ (self, f, g):
        """
        Register the function, f, which takes a numpy array a and returns a scalar
        function value f, and its gradient g, which takes a numpy array and returns
        a numpy array with gradients in each direction.
        """
        self.f = f
        self.g = g
        pass

    def optimize(self, startPoint=0, epsilon=1e-5, maxIterations=100):
        """
        Performs BFGS optimization until either minimum tolerenace (epsilon) or the
        maxIterations is reached. Tolerence is defined as the norm of the gradient.
        """
        n = len(startPoint)
        alpha = 1
        Hk = numpy.eye(n)
        I = numpy.eye(n)
        k = 0
        xk = startPoint
        gk = self.g(xk)
        
        while 1:
            # Compute the norm of the gradient.
            gradNorm = numpy.sqrt(numpy.dot(gk, gk))

            # Display the function value for the current iteration.
            fk = f(xk)
            print "%d: fval = %d, norm = %f" % (k, fk, gradNorm)           
            
            # Termination based on tolerenace criterion.
            if (gradNorm <= epsilon):
                print "Terminating: Tolerence %f (fval = %f, norm = %f)"\
                    % (epsilon, fk, gradNorm)
                return {'optimalPoint':xk, 'functVal':fk}

            # Termination due to maximum iterations.
            if (k > maxIterations):
                print "Terminating: Max iterations %d (fval = %f, norm = %f)" \
                    % (i, fk, gradNorm) 
                return {'optimalPoint':xk, 'functVal':fk}

            # Computing the search direction.
            pk = -numpy.dot(Hk, gk)
            sk = alpha * pk
            xk1 = xk + sk
            gk1 = self.g(xk1)
            yk = gk1 - gk

            # Computing Hk1.
            rhok = 1.0 / numpy.dot(yk, sk)
            A = I - (rhok * numpy.outer(sk, yk))
            B = rhok * numpy.outer(sk, sk)
            Hk = numpy.dot(numpy.dot(A, Hk), A) + B

            # Update the variables for the next iteration.
            xk = xk1
            gk = gk1
            k += 1
            pass            
        pass
    pass


def example():
    """
    Illustrates the use of the BFGS class by minimizing the following function.
    f(x1, x2) = (x1 - 1)**2 + (x2 + 3)**2
    The minimum point is (1,-3) and the value of f at this point is zero.
    """
    Optimizer = BFGS(f, g)
    startPoint = 100 * numpy.ones(2);
    res = Optimizer.optimize(startPoint,
                             epsilon=1e-5,
                             maxIterations=10)
    print res
    pass

def f(x):
    """
    Function for the example.
    """
    return ((x[0] - 1) ** 2) + ((x[1] + 3) ** 2)

def g(x):
    """
    Gradient of the function for the example.
    """
    y = numpy.zeros(2)
    y[0] = 2 * (x[0] - 1)
    y[1] = 2 * (x[1] + 3)
    return y

if __name__ == '__main__':
    example()
