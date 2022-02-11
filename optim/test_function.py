import numpy as np

from optim.core.space import IntervalProduct

class TestFunction:
    def __init__(self):
        self.n = 1
        self.x_star = None
        self.domain = None
    def objective(self ,x):
        return 0.
    def __call__(self,x):
        # evaluation of a scalar
        if len(x.shape)==0 and self.n==1:
            return self.objective(x)
        # evalutation of a single point
        if self.n==x.shape[-1] and len(x.shape)==1:
            return self.objective(x)
        return np.fromiter([self(xi) for xi in x], dtype=np.float64)

class Rastrigin(TestFunction):
    def __init__(self, A=10., n=1):
        self.A=A
        self.n=n
        self.x_star = np.zeros(self.n)
        self.domain = IntervalProduct([[-5.12, 5.12] for _ in range(self.n)])
    def objective(self,x):
        return self.A*self.n + np.sum(x**2 - self.A*np.cos(2.*np.pi*x))

class Ackley(TestFunction):
    def __init__(self, n=None):
        self.n = 2
        self.x_star = np.zeros(2)
        self.domain = IntervalProduct([[-5., 5.] for _ in range(self.n)])

    def objective(self, x):
        return -20. * np.exp(-.2* np.sqrt(.5*np.sum(x**2))) - np.exp(.5*np.sum(np.cos(2*np.pi*x))) + np.exp(1) + 20.

class Sphere(TestFunction):
    def __init__(self, n=1):
        self.n=n
        self.domain=[-100,100]
        self.x_star = np.zeros(n)
        self.domain = IntervalProduct([[-100,100] for _ in range(self.n)])
    def objective(self, x):
        return np.sum(x**2)

class Rosenbrock(TestFunction):
    def __init__(self, n=2):
        self.n=n
        self.x_star = np.ones(n)
        self.domain = IntervalProduct([[-100,100] for _ in range(self.n)])
    def objective(self, x):
        return np.sum([100.*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(self.n-1)])

class Beale(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([3,.5])
        self.domain = IntervalProduct([[-4.5,4.5] for _ in range(self.n)])

    def objective(self, x):
        return (1.5-x[0]+np.prod(x))**2 + (2.25-x[0]+x[0]*(x[1]**2))**2 + (2.625-x[0]+x[0]*(x[1]**3))**2

class GoldsteinPrice(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([0,-1])
        self.domain = IntervalProduct([[-2,2] for _ in range(self.n)])

    def objective(self, x):
        return (1+((np.sum(x)+1)**2) * (19 - 14*x[0]+3*x[0]**2-14*x[1]+6*np.prod(x)+3*x[1]**2))*(30+((2*x[0] - 3*x[1])**2)*(18-32*x[0]+(12*x[0]**2)+48*x[1]-36*np.prod(x)+27*x[1]**2))

class Booth(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([1,3])
        self.domain = IntervalProduct([[-10,10] for _ in range(self.n)])

    def objective(self, x):
        return (x.dot([1,2])-7)**2 + (x.dot([2,1])-5)**2

class Bukin6(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([-10,1])
        self.domain = IntervalProduct([[-15,-5],[-3,3]])

    def objective(self, x):
        return 100*np.sqrt(np.abs(x[1] - .01*x[0]**2)) + .01*np.abs(x[0]+10)

class Matyas(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.zeros(2)
        self.domain = IntervalProduct([[-10,10] for _ in range(self.n)])

    def objective(self, x):
        return .26*np.sum(x**2) - .48*np.prod(x)

class Levi13(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.ones(2)
        self.domain = IntervalProduct([[-10,10] for _ in range(self.n)])
    def objective(self, x):
        return np.sin(3*np.pi*x[0])**2 + ((x[0]-1)**2)*(1+np.sin(3*np.pi*x[1])**2)+((x[1]**2 - 1)**2)*(1+np.sin(2*np.pi*x[1])**2)

class Himmelblau(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([[3,2],
                                [-2.805118, 3.131312],
                                [-3.779310, -3.283186],
                                [3.584428, -1.848126]])

        self.domain = IntervalProduct([[-5,5] for _ in range(self.n)])
    def objective(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

class ThreeHumpCamel(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.zeros(self.n)
        self.domain = IntervalProduct([[-5,5] for _ in range(self.n)])
    def objective(self, x):
        return 2*x[0]**2 - 1.05*x[0]**4 + ((x[0]**6)/6.) + np.prod(x) + x[1]**2

class Easom(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.pi*np.ones(self.n)
        self.domain = IntervalProduct([[-100,100] for _ in range(self.n)])
    def objective(self, x):
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-np.sum((x-np.pi)**2))

class CrossInTray(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([[1.34941,-1.34941],
                                [1.34941,1.34941],
                                [-1.34941,1.34941],
                                [-1.34941,-1.34941]])
        self.domain = IntervalProduct([[-10,10] for _ in range(self.n)])
    def objective(self, x):
        return -.0001 * (np.abs(np.sin(x[0])*np.sin(x[1])*np.exp(np.abs(100.-np.sqrt(np.sum(x**2))/np.pi)))+1)**.1

class Eggholder(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([512,404.2319])
        self.domain = IntervalProduct([[-512,512] for _ in range(self.n)])
    def objective(self, x):
        return -(x[1]+47)*np.sin(np.sqrt(np.abs(x.dot([.5,1])+47))) - x[0]*np.sin(np.sqrt(np.abs(np.diff(x) - 47.)))

class HolderTable(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([[8.05502,9.66459],
                                [-8.05502,9.66459],
                                [8.05502,-9.66459],
                                [-8.05502,-9.66459]])
 
        self.domain = IntervalProduct([[-10,10] for _ in range(self.n)])
    def objective(self, x):
        return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1.-np.sqrt(np.sum(x**2))/np.pi)))

class McCormick(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([-.54719,-1.54719])
        self.domain = IntervalProduct([[-1.5,4],[-3,4]])
    def objective(self, x):
        return np.sin(np.sum(x)) + np.diff(x)**2 + x.dot([-1.5,2.5]) + 1

class Schaffer2(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.zeros(self.n)
        self.domain = IntervalProduct([[-100,100] for _ in range(self.n)])
    def objective(self, x):
        return .5 + (np.sin(np.diff(x**2))**2 - .5)/((1.+.001*np.sum(x**2))**2)

class Schaffer4(TestFunction):
    def __init__(self, n=None):
        self.n=2
        self.x_star = np.array([[0.,1.25313],[0.,-1.25313]])
        self.domain = IntervalProduct([[-100,100] for _ in range(self.n)])
    def objective(self, x):
        return .5 + (np.cos(np.sin(np.abs(np.diff(x**2))))**2 - .5)/((1.+.001*np.sum(x**2))**2)

class StyblinskiTang(TestFunction):
    def __init__(self, n=1):
        self.n=n
        self.x_star = -2.903534*np.ones(self.n)
        self.domain = IntervalProduct([[-5,5] for _ in range(self.n)])
    def objective(self, x):
        return np.sum(x**4 - 16*x**2 + 5*x) / 2.





test_functions = [Rastrigin,
                  Ackley,
                  Sphere,
                  Rosenbrock,
                  Beale,
                  GoldsteinPrice,
                  Booth,
                  Bukin6,
                  Matyas,
                  Levi13,
                  Himmelblau,
                  ThreeHumpCamel,
                  Easom,
                  CrossInTray,
                  Eggholder,
                  HolderTable,
                  McCormick,
                  Schaffer2,
                  Schaffer4,
                  StyblinskiTang,
                  ]

if __name__=="__main__":
    rastrigin = Rastrigin()
    print("rastrigin(x_star) = {}".format(rastrigin(rastrigin.x_star)))

    x = rastrigin.domain.meshgrid(1000)
    y = rastrigin(x)


    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(x,y)
    #plt.show()
    print()
    
    import matplotlib.pyplot as plt
    columns = 4
    N = len(test_functions)
    rows = int(N/columns)
    fig, axes = plt.subplots(rows,columns, figsize=(columns*5,5*rows))
    for i in range(len(test_functions)):
        cls = test_functions[i]
        print(cls.__name__)
        func = cls(n=2)
        print("\tx_star = {}".format(func.x_star))
        print("\t{}(x_star) = {}".format(cls.__name__,func(func.x_star)))
        print()

        x,y = func.domain.meshgrid(100)
        z = np.zeros(x.shape)
        for j in range(x.shape[0]):
            for k in range(y.shape[1]):
                xx  = np.array([x[j,k],y[j,k]])
                t = func(xx)
                z[j,k]=t

        axes[int(i/columns),i%columns].contour(x,y,z,100)
        axes[int(i/columns),i%columns].scatter(func.x_star[0], func.x_star[1], c="g")
        axes[int(i/columns),i%columns].set_title(cls.__name__)
    plt.tight_layout()
    plt.savefig("test_function_contours.png")

