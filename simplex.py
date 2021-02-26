import numpy as np
from fractions import Fraction
from enum import Enum

def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])
def exercise2_5(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])
def exercise2_6(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])
def exercise2_7(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def random_lp(n,m,sigma=10): return np.round(sigma*np.random.randn(n)),np.round(sigma*np.random.randn(m,n)),np.round(sigma*np.abs(np.random.randn(m)))

class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.
    
    def __init__(self,c,A,b,dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A' 
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m,n = A.shape
        self.dtype=dtype
        if dtype == int:
            self.lastpivot=1
        if dtype in [int,Fraction]:
            dtype=object
            if c is not None:
                c=np.array(c,np.object)
            A=np.array(A,np.object)
            b=np.array(b,np.object)
        self.C = np.empty([m+1,n+1+(c is None)],dtype=dtype)
        self.C[0,0]=self.dtype(0)
        if c is None:
            self.C[0,1:]=self.dtype(0)
            self.C[0,n+1]=self.dtype(-1)
            self.C[1:,n+1]=self.dtype(1)
        else:
            for j in range(0,n):
                self.C[0,j+1]=self.dtype(c[j])
        for i in range(0,m):
            self.C[i+1,0]=self.dtype(b[i])
            for j in range(0,n):
                self.C[i+1,j+1]=self.dtype(-A[i,j])
        self.N = np.array(range(1,n+1+(c is None)))
        self.B = np.array(range(n+1+(c is None),n+1+(c is None)+m))
        self.varnames=np.empty(n+1+(c is None)+m,dtype=object)
        self.varnames[0]='z'
        for i in range(1,n+1):
            self.varnames[i]='x{}'.format(i)
        if c is None:
            self.varnames[n+1]='x0'
        for i in range(n+1,n+m+1):
            self.varnames[i+(c is None)]='x{}'.format(i)

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m,n = self.C.shape
        varlen = len(max(self.varnames,key=len))
        coeflen = 0
        for i in range(0,m):
            coeflen=max(coeflen,len(str(self.C[i,0])))
            for j in range(1,n):
                coeflen=max(coeflen,len(str(abs(self.C[i,j]))))
        tmp=[]
        if self.dtype==int and self.lastpivot!=1:
            tmp.append(str(self.lastpivot))
            tmp.append('*')
        tmp.append('{} = '.format(self.varnames[0]).rjust(varlen+3))
        tmp.append(str(self.C[0,0]).rjust(coeflen))
        for j in range(0,n-1):
            tmp.append(' + ' if self.C[0,j+1]>0 else ' - ')
            tmp.append(str(abs(self.C[0,j+1])).rjust(coeflen))
            tmp.append('*')
            tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0,m-1):
            tmp.append('\n')
            if self.dtype==int and self.lastpivot!=1:
                tmp.append(str(self.lastpivot))
                tmp.append('*')
            tmp.append('{} = '.format(self.varnames[self.B[i]]).rjust(varlen+3))
            tmp.append(str(self.C[i+1,0]).rjust(coeflen))
            for j in range(0,n-1):
                tmp.append(' + ' if self.C[i+1,j+1]>0 else ' - ')
                tmp.append(str(abs(self.C[i+1,j+1])).rjust(coeflen))
                tmp.append('*')
                tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        return ''.join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m,n = self.C.shape
        if self.dtype==int:
            x_dtype=Fraction
        else:
            x_dtype=self.dtype
        x = np.empty(n-1,x_dtype)
        x[:] = x_dtype(0)
        for i in range (0,m-1):
            if self.B[i]<n:
                if self.dtype==int:
                    x[self.B[i]-1]=Fraction(self.C[i+1,0],self.lastpivot)
                else:
                    x[self.B[i]-1]=self.C[i+1,0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype==int:
            return Fraction(self.C[0,0],self.lastpivot)
        else:
            return self.C[0,0]

    def pivot(self,k,l):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int

        # save pivot coefficient
        a = self.C[l+1,k+1]
        # TODO
        pass


class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3

#D.B = array of indices of basic variables
#D.N = array of indices of non-basic variables
#A = coefficients of variables in constraints
#c = coefficients of objective function
#b = values of basic variables

def bland(D,eps):
    print(D)
    obj_coeff = D.C[0, 1:]
    constraint_coeff = D.C[1:, 1:]
    constraint_constants = D.C[1:, 0]
    entering=None
    for index in range(len(obj_coeff)):
        temp = obj_coeff[index]
        if( temp > 0):
            entering = index
    
    if(entering is None):
        return None, None
    
    leaving=None
    tightest_ratio = np.inf 
    candidates =[]
    for index in range(constraint_coeff.shape(1)):
        coefficient = D.A[index][entering] 
        ratio = ratio(coefficient, D.B[index])
        
        if ( ratio < tightest_ratio & ratio >= 0):
            tightest_ratio = ratio
            candidates = []
            candidates.append(constraint_constants[index])
            
        elif (ratio == tightest_ratio):
            candidates.append(constraint_constants[index])
    
    leaving = candidates.min()
    return entering, leaving
    
def ratio(x, y, eps):
    temp = x/y
    if np.abs(temp) < eps:
        temp = 0 #eps>=0 is such that numbers in the closed interval [-eps,eps]
    return temp
    # x1+w2+x3
       
    # B(0)=x1
    # B(1)=w2

    # N(1)= x1 
    # N(2)= x2
    # N(n+1)=w1 

    #iterate over variables in objective func and store vars with positive coeffienct
 

    # Assumes a feasible dictionary D and finds entering and leaving
    # variables according to Bland's rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
       
    # k=l=None
    # # TODO
    # return k,l
    


def largest_coefficient(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k=l=None
    # TODO
    return k,l

def largest_increase(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k=l=None
    # TODO
    return k,l

def lp_solve(c,A,b,dtype=Fraction,eps=0,pivotrule=lambda D: bland(D,eps=0),verbose=False):
    # Simplex algorithm
    #    
    # Input is LP in standard form given by vectors and matrices
    # c,A,b.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0.
    #
    # pivotrule is a rule used for pivoting. Cycling is prevented by
    # switching to Bland's rule as needed.
    #
    # If verbose is True it outputs possible useful information about
    # the execution, e.g. the sequence of pivot operations
    # performed. Nothing is required.
    #
    # If LP is infeasible the return value is LPResult.INFEASIBLE,None
    #
    # If LP is unbounded the return value is LPResult.UNBOUNDED,None
    #
    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,D, where D is an optimal dictionary.

    #TODO
    return None,None
  
def run_examples():

    #ratiotest
    ratio_test = ratio(0.00001,1, 1e-5)
    print("ratio Tets")
    print(ratio_test)
    # Example 1
    c,A,b = example1()
    D=Dictionary(c,A,b)
    str(D)
    print(D)
    bland(D,1e-3)
    print(D.B)
    print(D.N)
    print(A)
    print(c)
    print(b)
    print('Example 1 with Fraction')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print()

    D=Dictionary(c,A,b,np.float64)
    print('Example 1 with np.float64')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print()

    # # Example 2
    # c,A,b = example2()
    # print('Example 2')
    # print('Auxillary dictionary')
    # D=Dictionary(None,A,b)
    # print(D)
    # print('x0 is entering and x4 leaving:')
    # D.pivot(2,1)
    # print(D)
    # print('x2 is entering and x3 leaving:')
    # D.pivot(1,0)
    # print(D)
    # print('x1 is entering and x0 leaving:')
    # D.pivot(0,1)
    # print(D)
    # print()

    # # Solve Example 1 using lp_solve
    # c,A,b = example1()
    # print('lp_solve Example 1:')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # # Solve Example 2 using lp_solve
    # c,A,b = example2()
    # print('lp_solve Example 2:')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # # Solve Exercise 2.5 using lp_solve
    # c,A,b = exercise2_5()
    # print('lp_solve Exercise 2.5:')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # # Solve Exercise 2.6 using lp_solve
    # c,A,b = exercise2_6()
    # print('lp_solve Exercise 2.6:')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # # Solve Exercise 2.7 using lp_solve
    # c,A,b = exercise2_7()
    # print('lp_solve Exercise 2.7:')
    # res,D=lp_solve(c,A,b)
    # print(res)
    # print(D)
    # print()

    # #Integer pivoting
    # c,A,b=example1()
    # D=Dictionary(c,A,b,int)
    # print('Example 1 with int')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()

    # c,A,b = integer_pivoting_example()
    # D=Dictionary(c,A,b,int)
    # print('Integer pivoting example from lecture')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x3 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x2 is entering and x4 leaving:')
    # D.pivot(1,1)
    # print(D)


run_examples();

