import numpy as np
from fractions import Fraction
from enum import Enum
import time
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import copy

def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])
def book_dual_example(): return np.array([-1,4]), np.array([[-2,-1],[-2,4],[-1,3]]), np.array([4,-8,-7])
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
        # swap index of entering and leaving var in index array
        leaving = self.B[l]
        entering = self.N[k]
        self.N[k] = leaving
        self.B[l] = entering 
        t0 = time.time()
        for row_index in range(self.C[:,0].size):  #subtract/add leaving var to all other rows
            if(row_index != l+1):
                leaving_value = self.C[row_index, k+1]
                leaving_ratio = ratio(leaving_value, a)
                self.C[row_index, :] = self.C[row_index, :] - leaving_ratio * self.C[l+1, :]
                self.C[row_index, k+1] = leaving_ratio
        self.C[l+1,k+1] = -self.C[l+1,k+1]/self.C[l+1,k+1]  #swap leaving and entering
        self.C[l+1,:] = self.C[l+1,:]/(-a)  #normalize by B[L] coefficient
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

    obj_coeff = D.C[0, 1:]
    constraint_coeff = D.C[1:, 1:]
    constraint_constants = D.C[1:, 0]
    entering=None
    index_of_entering= np.inf
    #find entering variable:
    for index in range(len(obj_coeff)):
        temp = obj_coeff[index]
        if( temp > 0 and D.N[index] < index_of_entering):
            index_of_entering = D.N[index]
            entering = index
    
    if(entering is None):
        return None, None
    
    leaving=None
    tightest_ratio = -np.inf 
    leaving_candidates = []
    constraint_coeff_entering = constraint_coeff[:,entering]
    #find leaving variable
    for index in range(constraint_coeff_entering.size):
        coefficient_entering = constraint_coeff_entering[index]
        constraint_constant = constraint_constants[index]
        constraint_ratio = ratio(constraint_constant, coefficient_entering, eps)
        if ( constraint_ratio > tightest_ratio and constraint_ratio < 0):
            tightest_ratio = constraint_ratio
            leaving_candidates = []
            leaving_candidates.append(D.B[index])
            
        elif (constraint_ratio == tightest_ratio):
            leaving_candidates.append(D.B[index])

    if(len(leaving_candidates) == 0):
        return entering, None

    best = min(leaving_candidates)
    
    for index in range(D.B.size):
        if (D.B[index] == best):
            leaving = index
    return entering, leaving
    
def ratio(x, y, eps=1e-5):
    if( y == 0):
        return 0
    temp = x/y
    if np.abs(temp) < eps:
        temp = 0 #eps>=0 is such that numbers in the closed interval [-eps,eps]
    return temp

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
    
    D = Dictionary(c, A, b)
    k,l = pivotrule(D)
    res = None, None
    while (k != None and l != None):
        D.pivot(k,l)
        k, l = pivotrule(D)
    if(k != None and l == None):
        res = LPResult.UNBOUNDED, None
    if(k == None):
        res = LPResult.OPTIMAL, D
    return res


def run_experiment_1phase_alg(no_of_iterations):

    iterations_results_float = []
    iterations_results_fraction = []
    
    for i in range(no_of_iterations):
        #The formulas for m and n produce numbers between 10 and 100.
        n = int(np.round(np.exp(np.log(50)*np.random.rand())+10))
        m = int(np.round(np.exp(np.log(50)*np.random.rand())+10))
        print("started running")
        c,A,b = random_lp(n,m)
        print("m is: "+ str(m))
        print("n is: "+ str(n))

        t0 = time.time()
        res = linprog(c,A,b)
        t1 = time.time()
        total_time_scipy = t1-t0
        print("finished SciPy in time:" + str(total_time_scipy))
        
        t0 = time.time()
        res_float, D = lp_solve(c,A,b, dtype = np.float64)
        t1 = time.time()
        total_time_float = t1-t0
        print("finished float64 in time:" + str(total_time_float))
        t2 = time.time()
        res_fraction, D = lp_solve(c,A,b, dtype = Fraction)
        t3 = time.time()   
        total_time_fraction = t3-t2
        print("finished Fraction in time:" + str(total_time_fraction))
        iterations_results_float.append((n,m,total_time_float, res_float))
        iterations_results_fraction.append((n,m,total_time_fraction, res_fraction))
    
    np.save("./results/iterations_results_float.npy",iterations_results_float, allow_pickle = True)
    np.save("./results/iterations_results_fraction.npy",iterations_results_fraction,  allow_pickle = True)

    
def print_experiment(array1, array2):
    fig=plt.figure()
    arr_n_plus_m = [_tuple[0]+_tuple[1] for _tuple in array1]
    arr1_time = [_tuple[2] for _tuple in array1]
    arr2_time = [_tuple[2] for _tuple in array2]
    plt.scatter(arr_n_plus_m, arr1_time, color='r', marker = "x", label ="fraction")
    plt.scatter(arr_n_plus_m, arr2_time, color='b', marker = "o", label ="float")
    plt.xlabel('n+m')
    plt.ylabel('Time')
    plt.title('Time experiment for n+m')
    plt.legend()
    plt.show()



def get_dual_dictionary(primal_dictionary):
    dual_dictionary = copy.deepcopy(primal_dictionary)
    dual_dictionary.C = -np.transpose(primal_dictionary.C)
    non_basic_size = primal_dictionary.N.size
    basic_size = primal_dictionary.B.size
    # dual_dictionary.B = np.zeros(primal_dictionary.N.size)
    _type = type(primal_dictionary.N[0])
    dual_B = np.empty(non_basic_size, dtype=_type)
    dual_N = np.empty(basic_size, dtype=_type)
    # dual_dictionary.N = np.zeros(primal_dictionary.B.size)
    for index in range(non_basic_size):
        if(primal_dictionary.N[index] <= non_basic_size):
            dual_B[index] = primal_dictionary.N[index] + basic_size
        else: dual_B[index] =  primal_dictionary.N[index] - non_basic_size

    for index in range(basic_size):
        if(primal_dictionary.B[index] <= non_basic_size):
            dual_N[index] = primal_dictionary.B[index] + basic_size
        else: dual_N[index] =  primal_dictionary.B[index] - non_basic_size
    dual_dictionary.B = dual_B
    dual_dictionary.N = dual_N
    return dual_dictionary
    


def phase1_alg(D, pivotrule=lambda D: bland(D,eps=0)):
    #insert new obj func and remeber old one
    original_obj_func = copy.deepcopy(D.C[0,1:])
    original_non_basic = copy.deepcopy(D.N)
    D.C[0, 1:] = -(np.ones(D.C[0, 1:].size))
    D.C[0,0] = 0

    #det dual og modified dictionary
    dual_D = get_dual_dictionary(D)

    #TODO: modify to use generic rule
    #use rule to find pivot indicies
    k,l = pivotrule(dual_D)
    while (k != None and l != None):
        dual_D.pivot(k,l)
        print(dual_D)
        k, l = pivotrule(dual_D)
    if(k != None and l == None):
        return LPResult.INFEASIBLE, None

    #get dual of dual = primal
    D = get_dual_dictionary(dual_D)

    # substitute the answer from feasable dictionary into obj function from original primal problem 
    new_obj_func = np.array([])
    variables_substituted = []
    #check for each basic var in feasable dual if it is in original obj func
    #yes -> add coeffecient times value in org obj func to new obj function
    for index in range(D.B.size):
        if D.B[index] in original_non_basic: 
            index_in_N = original_non_basic.tolist().index(D.B[index])
            if new_obj_func.size != 0:
                new_obj_func = new_obj_func + original_obj_func[index_in_N]*D.C[index+1,:]
            else: new_obj_func = original_obj_func[index_in_N]*D.C[index +1,:]
            variables_substituted.append(index_in_N+1)

    print(new_obj_func)
    #iterate over all original nonbasic - If not substituted, add coeff to coeff in new obj func
    for index in range(original_non_basic.size):
        if not (original_non_basic[index] in variables_substituted): #
            index_new_obj = D.N.tolist().index(original_non_basic[index])+1
            new_obj_func[index_new_obj] = new_obj_func[index_new_obj] + original_obj_func[index]
    
    # assign new obj function after substituting
    D.C[0,:] = new_obj_func
    return None, D



def run_examples():
    #arr1 = np.load("./results/iterations_results_fraction.npy",  allow_pickle = True)
    #arr2 = np.load("./results/iterations_results_float.npy",  allow_pickle = True)
    #print_experiment(arr1,arr2)
    # c,A,b = exercise2_7()
    # d = Dictionary(c,A,b)
    # phase1_alg(d)

    c,A,b = book_dual_example()
    d = Dictionary(c,A,b)
    print("Primal:")
    print(d)
    phase1_alg(d)

    return
    #run_experiment_1phase_alg(100)
    #ratiotest
    #Example 1
    # c,A,b = example1()
    # D=Dictionary(c,A,b)
    # print('Example 1 with Fraction')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()

    # D=Dictionary(c,A,b,np.float64)
    # print('Example 1 with np.float64')
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    # D.pivot(0,0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()

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

    # Solve Example 1 using lp_solve
    c,A,b = exercise2_7()
    d = Dictionary(c,A,b)
    print('lp_solve Example 1:')
    res,D=lp_solve(c,A,b)
    get_dual_dictionary(d)
    print(res)
    print(D)
    print()

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

