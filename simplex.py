import numpy as np
from fractions import Fraction
from enum import Enum
import time
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import copy
import math

def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example1_1(): return np.array([4,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])
def book_dual_example(): return np.array([-1,4]), np.array([[-2,-1],[-2,4],[-1,3]]), np.array([4,-8,-7])
def exercise2_5(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])
def exercise2_6(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])
def exercise2_7(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def failing_example(): return np.array([-14,5,8,2,-5]),np.array([[-1,1,13,-15,-21],[14,8,15,-12,-6],
       [ 16,-2,14,-13,10],
       [-10,-10,7,-2,-1]]), np.array([11,0,14,2])
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

    def pivot(self,k,l, eps):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        
        # save pivot coefficient
        a = self.C[l+1,k+1]
        # swap index of entering and leaving var in index array
        leaving = self.B[l]
        entering = self.N[k]
        self.N[k] = leaving
        self.B[l] = entering 

        if(self.dtype == int):
            for row_index in range(self.C[:,0].size): # multiply each row by entering coeff
                if(row_index != l+1):
                    self.C[row_index, :] = -a * self.C[row_index, :] 
    
        for row_index in range(self.C[:,0].size):  #subtract/add leaving var to all other rows
            if(row_index != l+1):
                leaving_value = self.C[row_index, k+1]
                if(self.dtype == int):
                    leaving_ratio = ratio_int(leaving_value, a, eps)
                else: leaving_ratio = ratio(leaving_value, a, eps)
                self.C[row_index, :] = self.C[row_index, :] - leaving_ratio * self.C[l+1, :]
                self.C[row_index, k+1] = leaving_ratio
        
        if(self.dtype != int):
            self.C[l+1,k+1] = -self.C[l+1,k+1]/self.C[l+1,k+1]  #swap leaving and entering
            self.C[l+1,:] = self.C[l+1,:]/(-a)  #normalize by B[L] coefficient
        else:
            self.C[l+1,k+1] = -self.lastpivot  
            for row_index in range(self.C[:,0].size):  #divide all rows, except the pivot row, by the previous negatuve pivot coefficient
                if(row_index != l+1):
                    self.C[row_index,k+1] = self.lastpivot * self.C[row_index,k+1] 
                    self.C[row_index, :] = self.C[row_index, :] // self.lastpivot   #  // is integer division
            self.lastpivot = -a
        pass

class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3

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
        if ( constraint_ratio > tightest_ratio and constraint_ratio <= 0 and coefficient_entering < 0):
            tightest_ratio = constraint_ratio
            leaving_candidates = []
            leaving_candidates.append(D.B[index])
            
        elif (constraint_ratio == tightest_ratio and constraint_ratio <= 0 and coefficient_entering < 0):
            leaving_candidates.append(D.B[index])

    if(len(leaving_candidates) == 0):
        return entering, None

    best = min(leaving_candidates)
    
    for index in range(D.B.size):
        if (D.B[index] == best):
            leaving = index
    return entering, leaving
    
def ratio(x, y, eps):
    if( y == 0):
        return 0
    temp = x/y
    if np.abs(temp) < eps:
        temp = 0 #eps>=0 is such that numbers in the closed interval [-eps,eps]
    return temp

def ratio_int(x, y, eps):
    if( y == 0):
        return 0
    temp = x//y
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
    largest = 0
    entering = None
    obj_func_coeffs = D.C[0, 1:]
    for i in range(obj_func_coeffs.size):
        coeff = obj_func_coeffs[i]
        if(coeff > largest):
            largest = coeff
            entering = i
    if(entering is None):
        return None, None
    leaving=None
    tightest_ratio = -np.inf 
    constraint_coeff_entering = D.C[1:,entering+1]
    constraint_constants = D.C[1:, 0]

    #find leaving variable
    for i in range(constraint_coeff_entering.size):
        coefficient_entering = constraint_coeff_entering[i]
        constraint_constant = constraint_constants[i]
        constraint_ratio = ratio(constraint_constant, coefficient_entering, eps)
        if ( constraint_ratio > tightest_ratio and constraint_ratio <= 0 and coefficient_entering < 0):  #Is last condition right???
            tightest_ratio = constraint_ratio
            leaving = i

    return entering, leaving


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

    #Find most restricting ratio for all leaving and multiply with coeffient in obj func
    obj_func_coeffs = D.C[0, 1:]
    leaving = None
    entering = None
    entering_increase = -np.inf
    for N_i in range(obj_func_coeffs.size):
        obj_entering_coeff = obj_func_coeffs[N_i] 
        tightest_ratio = -np.inf
        leaving_index = None
        if(obj_entering_coeff > 0):
            if(entering == None):
                entering = N_i  
            for B_i in range(D.C[1:, N_i+1].size):
                constraint_constant = D.C[B_i+1,0]
                coefficient_entering = D.C[B_i+1,N_i+1]
                constraint_ratio = ratio(constraint_constant, coefficient_entering, eps)
                if (constraint_ratio > tightest_ratio and constraint_ratio <= 0 and coefficient_entering < 0):
                    tightest_ratio = constraint_ratio
                    leaving_index = B_i
        # if(obj_entering_coeff == 0):
        #     entering_inc = 0
        # else:
        entering_inc = abs(tightest_ratio)*obj_entering_coeff
        if(entering_inc > entering_increase):
            entering_increase = entering_inc
            entering = N_i
            leaving = leaving_index
    return entering, leaving

def check_if_degenerate(D):
    coefficient_col = D.C[1:,0]
    contains_0 = 0 in coefficient_col
    return contains_0

def find_and_perform_pivots(D, eps, pivotrule=lambda D, eps: bland(D,eps)):
    is_degenerate = check_if_degenerate(D)
    degenerate_count = 0
    if(is_degenerate):
        degenerate_count += 1   
    k,l = pivotrule(D, eps)
    res = None, None
    while (k != None and l != None):
        D.pivot(k,l, eps)
        is_degenerate = check_if_degenerate(D)
        if(is_degenerate):
            degenerate_count += 1
        else:
            degenerate_count = 0
        if (degenerate_count >= (D.B.size)):
            k, l = bland(D, eps)
        else:
            k, l = pivotrule(D, eps)
    if(k != None and l == None):
        res = LPResult.UNBOUNDED, None
    if(k == None):
        res = LPResult.OPTIMAL, D
    return res

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
    
def insert_new_obj_func(D, original_obj_func, original_non_basic):
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

    # print(new_obj_func)
    #iterate over all original nonbasic - If not substituted, add coeff to coeff in new obj func
    for index in range(original_non_basic.size):
        if not (original_non_basic[index] in variables_substituted): #
            index_new_obj = D.N.tolist().index(original_non_basic[index])+1
            new_obj_func[index_new_obj] = new_obj_func[index_new_obj] + original_obj_func[index]

    D.C[0,:] = new_obj_func
    return D
    

def phase1_alg(D, eps, pivotrule=lambda D, eps: bland(D,eps)):
    #insert new obj func and remeber old one
    original_obj_func = copy.deepcopy(D.C[0,1:])
    original_non_basic = copy.deepcopy(D.N)
    D.C[0, 1:] = -(np.ones(D.C[0, 1:].size))
    D.C[0,0] = 0

    #det dual of modified dictionary
    dual_D = get_dual_dictionary(D)

    res, optimal_dual = find_and_perform_pivots(dual_D, eps, pivotrule)
    #If dual is unbounded, then primal is infeasible
    if(res == LPResult.UNBOUNDED):
        return res, None

    #get dual of dual = primal, and insert new obj_func
    D = get_dual_dictionary(optimal_dual)
    D = insert_new_obj_func(D, original_obj_func, original_non_basic)
    return res, D



def lp_solve(c,A,b,dtype=Fraction,eps=1e-5,pivotrule=lambda D, eps: bland(D,eps),verbose=False):
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

    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,D, where D is an optimal dictionary.
    
    D = Dictionary(c, A, b, dtype = dtype)
    constraint_constants = D.C[1:, 0]
    res = None
    no_of_negative_constants1 = sum(n < 0 for n in constraint_constants)
    is_infeasible = 0 < no_of_negative_constants1
    if(is_infeasible):
        res_phase1, D = phase1_alg(D, eps, pivotrule)
        if(res_phase1 is LPResult.UNBOUNDED):
            return LPResult.INFEASIBLE, D
    res = find_and_perform_pivots(D, eps, pivotrule)
    return res

def run_examples():
    test_integer_pivot1()
    floating_rounding_test()
    #run_experiment_1(100)
    #print_experiment_1()
    #run_experiment_2(200)
    #run_experiment_inter_vs_fraction_pivot(200)
    #print_integer_experiment()
    test_integer_pivot()
    #print_experiment_2()
    test_correctness_of_algorithms(10)
    test_largest_coeff_example1()
    test_largest_increase_example1()
    test_2_6_phase()
    test_2_5_phase()
    test_2_7_phase()
    test_book_example()
    test_failing()
    #run_experiment_1phase_alg(100)
    return

def floating_rounding_test():
    res = ratio(0.000009,1, 1e-5)
    assert(res == 0)
    res = ratio(0.00001,1, 1e-5)
    assert(res == 1.0e-5)



def test_integer_pivot1():
    c,A,b = integer_pivoting_example()
    d = Dictionary(c,A,b, dtype= int)
    entering, leaving = largest_coefficient(d, 1e-5)
    #print(d)
    d.pivot(entering, leaving, 1e-5)
    entering, leaving = largest_coefficient(d, 1e-5)
    #print(d)
    d.pivot(entering, leaving, 1e-5)
    #print(d)
    
def test_integer_pivot():
    c,A,b = integer_pivoting_example()
    res_int, D_int = lp_solve(c,A,b, dtype=int)
    # print(res_int)
    # print(D_int)
    res_float, D_float = lp_solve(c,A,b)
    # print(res_float)
    # print(D_float)
    
def test_largest_coeff_example1():
    print("Test of choosing largest entering and smallest ratio leaving")
    c,A,b = example1()
    d = Dictionary(c,A,b)
    entering, leaving = largest_coefficient(d, 1e-5)
    assert(entering == 0)
    assert(leaving == 0)

def test_largest_increase_example1():
    print("Test of choosing entering that increases z the most")
    c,A,b = exercise2_5()
    # assert(entering == 2)
    # assert(leaving == 2)
    res = lp_solve(c,A,b, eps = 1e-5, dtype=np.float64, pivotrule=lambda D, eps: largest_increase(D,eps))
    
def test_infeasible_primal_unbounded_dual():
    print("infeasible test of ex. 2.5")
    c,A,b = exercise2_5() # unbounded
    d = Dictionary(c,A,b)
    dual = get_dual_dictionary(d)  #dual of undbounded is infeasable
    res, D = phase1_alg(dual)
    assert(res == LPResult.UNBOUNDED)
    assert(D == None)
    print("test_infeasable_primal_unbounded_dual SUCCESFUL")

def test_2_5_phase():
    c,A,b = exercise2_5() #optimal
    res, D = lp_solve(c,A,b)
    res_linprog = linprog(-c,A,b)
    assert(res_linprog.status == 0) 
    assert(res == LPResult.OPTIMAL)

def test_2_6_phase():
    c,A,b = exercise2_6() # infeasible
    res, D = lp_solve(c,A,b, eps = 1e-5, dtype=np.float64, pivotrule=lambda D, eps: largest_coefficient(D, eps))
    res_linprog = linprog(-c,A,b, method="simplex")
    print(res)
    print(res_linprog.status)
    assert(res_linprog.status == 2) 
    assert(res == LPResult.INFEASIBLE)

def test_2_7_phase():
    c,A,b = exercise2_7() # unbounded
    res, D = lp_solve(c,A,b)
    res_linprog = linprog(-c,A,b)
    assert(res_linprog.status == 3) 
    assert(res == LPResult.UNBOUNDED)

def test_book_example():
    c,A,b = book_dual_example() # unbounded
    res, D = lp_solve(c,A,b)
    res_linprog = linprog(-c,A,b)
    assert(res_linprog.status == 3) 
    assert(res == LPResult.UNBOUNDED)

def test_failing():
    c,A,b = failing_example()
    res, D = lp_solve(c,A,b, eps = 1e-5, dtype=np.float64, pivotrule=lambda D, eps: largest_coefficient(D, eps))
    res_linprog = linprog(-c,A,b, method="simplex")
    assert(res == LPResult.UNBOUNDED)
    assert(res_linprog.status == 3)


def test_correctness_of_algorithms(number_of_iterations):
    np.random.seed(1)
    for i in range(number_of_iterations):
        print(i)
        n = int(np.round(np.exp(np.log(50)*np.random.rand()) + 10))
        m = int(np.round(np.exp(np.log(50)*np.random.rand()) + 10))
        c,A,b = random_lp(n,m)
        res_bland, D_bland = lp_solve(c,A,b, eps = 1e-5)
        res_largest_coeff, D_largest_coeff = lp_solve(c,A,b, eps = 1e-5, pivotrule= lambda D, eps: largest_coefficient(D, eps))
        res_largest_increase, D_largest_increase = lp_solve(c,A,b, eps = 1e-5, pivotrule= lambda D, eps: largest_increase(D, eps))
        res_bland_int, D_bland_int = lp_solve(c,A,b, eps = 1e-5, pivotrule= lambda D, eps: largest_increase(D, eps), dtype=int)
        res_linprog = linprog(-c,A,b, method="simplex")
        try:
            assert(res_bland == res_largest_coeff == res_largest_increase==res_bland_int)
        except:
            print(res_linprog.status)
            print(res_bland)
            print(res_largest_coeff)
            print(res_largest_increase)
            print(res_bland_int)
        if(res_bland == LPResult.OPTIMAL):
            try:
                assert(D_bland.value() == D_largest_coeff.value() == D_largest_increase.value() == D_bland_int.value())
            except: 
                print(D_bland.value())
                print(D_largest_coeff.value())
                print(D_largest_increase.value())
                print(D_bland_int.value())
        if(res_bland == LPResult.OPTIMAL):
            assert(res_linprog.status == 0)
        if(res_bland == LPResult.UNBOUNDED):
            try: 
                assert(res_linprog.status == 3 or res_linprog == 2)
            except: 
                None
        if(res_bland == LPResult.INFEASIBLE):
            assert(res_linprog.status == 2)
    print("it is amazing how this just works")

def run_experiment_1(no_of_iterations):
    iterations_results_float = []
    iterations_results_fraction = []
    iterations_results_scipy = []
    
    np.random.seed(1)
    for i in range(no_of_iterations):
        print(i)
        #The formulas for m and n produce numbers between 10 and 100.
        n = int(np.round(np.exp(np.log(140)*np.random.rand())+10))
        m = int(np.round(np.exp(np.log(140)*np.random.rand())+10))
        print("started running")
        c,A,b = random_lp(n,m)
        print("m is: "+ str(m))
        print("n is: "+ str(n))
        
        t0 = time.time()
        res_float, D = lp_solve(c,A,b, eps = 1e-5, dtype = np.float64)
        t1 = time.time()
        total_time_float = t1-t0
        t2 = time.time()
        res_fraction, D = lp_solve(c,A,b, eps = 1e-5, dtype = Fraction)
        t3 = time.time()   
        total_time_fraction = t3-t2
        iterations_results_float.append((n,m,total_time_float, res_float))
        iterations_results_fraction.append((n,m,total_time_fraction, res_fraction))
    
    np.save("./results/iterations_results_float.npy",iterations_results_float, allow_pickle = True)
    np.save("./results/iterations_results_fraction.npy",iterations_results_fraction,  allow_pickle = True)

def run_experiment_2(no_of_iterations):
    iterations_results_largest_increase_float = []
    iterations_results_largest_increase_fraction = []
    iterations_results_largest_coeff_float = []
    iterations_results_largest_coeff_fraction = []
    iterations_results_scipy = []
    eps = 1e-5
    counter = 0
    np.random.seed(1)
    for i in range(no_of_iterations):
        #The formulas for m and n produce numbers between 10 and 100.
        n = int(np.round(10*np.exp(np.log(20)*np.random.rand())))
        m = int(np.round(10*np.exp(np.log(20)*np.random.rand())))
        print("n: " + str(n) + " m: " + str(m))
        c,A,b = random_lp(n,m)

        t0 = time.time()
        try:
            res = linprog(-c,A,b)
        except:
            None
        t1 = time.time()
        total_time_scipy = t1-t0
        iterations_results_scipy.append((n,m, total_time_scipy))
        
        t0 = time.time()
        res_float, D = lp_solve(c,A,b, eps = 1e-5, dtype = np.float64, pivotrule = lambda D, eps: largest_coefficient(D, eps))
        t1 = time.time()
        total_time_float = t1-t0
        t2 = time.time()
        res_fraction, D = lp_solve(c,A,b, eps = 1e-5, dtype = Fraction, pivotrule = lambda D, eps: largest_coefficient(D, eps))
        t3 = time.time()   
        total_time_fraction = t3-t2
        iterations_results_largest_coeff_float.append((n,m,total_time_float, res_float))
        iterations_results_largest_coeff_fraction.append((n,m,total_time_fraction, res_fraction))

        t0 = time.time()
        res_float, D = lp_solve(c,A,b, eps = 1e-5, dtype = np.float64, pivotrule = lambda D, eps: largest_increase(D, eps))
        t1 = time.time()
        total_time_float = t1-t0
        t2 = time.time()
        res_fraction, D = lp_solve(c,A,b, eps = 1e-5, dtype = Fraction, pivotrule = lambda D, eps: largest_increase(D, eps))
        t3 = time.time()   
        total_time_fraction = t3-t2
        iterations_results_largest_increase_float.append((n,m,total_time_float, res_float))
        iterations_results_largest_increase_fraction.append((n,m,total_time_fraction, res_fraction))
        print(counter)
        counter+=1

    np.save("./results/iterations_results_largest_coefficient_float.npy",iterations_results_largest_coeff_float, allow_pickle = True)
    np.save("./results/iterations_results_largest_coefficient_fraction.npy",iterations_results_largest_coeff_fraction, allow_pickle = True)

    np.save("./results/iterations_results_largest_increase_float.npy",iterations_results_largest_increase_float,  allow_pickle = True)
    np.save("./results/iterations_results_largest_increase_fraction.npy",iterations_results_largest_increase_fraction,  allow_pickle = True)
    
    np.save("./results/iterations_results_scipy.npy",iterations_results_scipy,  allow_pickle = True)


def run_experiment_inter_vs_fraction_pivot(no_of_iterations):
    iterations_results_largest_increase_int = []
    iterations_results_largest_increase_fraction = []
    counter = 0
    np.random.seed(1)
    for i in range(no_of_iterations):
        #The formulas for m and n produce numbers between 10 and 100.
        n = int(np.round(np.exp(np.log(140)*np.random.rand())+10))
        m = int(np.round(np.exp(np.log(140)*np.random.rand())+10))
        print("n: " + str(n) + " m: " + str(m))
        c,A,b = random_lp(n,m)
        
        t0 = time.time()
        res_int, D = lp_solve(c,A,b,eps = 1e-5, dtype = int, pivotrule = lambda D, eps: largest_coefficient(D, eps))
        t1 = time.time()
        total_time_int = t1-t0
        t2 = time.time()
        res_fraction, D = lp_solve(c,A,b,eps = 1e-5, dtype = Fraction, pivotrule = lambda D, eps: largest_coefficient(D, eps))
        t3 = time.time()   
        total_time_fraction = t3-t2

        iterations_results_largest_increase_int.append((n,m,total_time_int, res_int))
        iterations_results_largest_increase_fraction.append((n, m, total_time_fraction, res_fraction))
        print(counter)
        counter +=1
    np.save("./results/integer/iterations_results_largest_increase_integer.npy",iterations_results_largest_increase_int, allow_pickle = True)
    np.save("./results/integer/iterations_results_largest_increase_fraction.npy",iterations_results_largest_increase_fraction,  allow_pickle = True)
    
def print_experiment_1():
    res_float = np.load("./results/iterations_results_float.npy", allow_pickle=True)
    res_fraction = np.load("./results/iterations_results_fraction.npy", allow_pickle=True)

    fig=plt.figure()
    arr_n_plus_m = [_tuple[0]+_tuple[1] for _tuple in res_float]
    arr_min_n_m = [min(_tuple[0], _tuple[1]) for _tuple in res_fraction]
    float_time = [_tuple[2] for _tuple in res_float]
    fraction_time = [_tuple[2] for _tuple in res_fraction]

    # plt.scatter(arr_n_plus_m, float_time, color='b', marker='.', label="float")
    # plt.scatter(arr_n_plus_m, fraction_time, color='r', marker='.', label="fraction")

    plt.scatter(arr_min_n_m, float_time, color='b', marker='.', label="float")
    plt.scatter(arr_min_n_m, fraction_time, color='r', marker='.', label="fraction")

    plt.scatter
    # plt.xlabel('n+m')
    plt.xlabel('min(n,m)')
    plt.ylabel('Time')
    plt.title('Time experiment for n+m')
    plt.legend()
    plt.show()


def print_experiment_2():
    largest_coeff_float = np.load("./results/iterations_results_largest_coefficient_float.npy", allow_pickle=True)
    largest_coeff_fraction= np.load("./results/iterations_results_largest_coefficient_fraction.npy", allow_pickle = True)
    largest_increase_float= np.load("./results/iterations_results_largest_increase_float.npy",allow_pickle = True)
    largest_increase_fraction= np.load("./results/iterations_results_largest_increase_fraction.npy",allow_pickle = True)
    scipy_res = np.load("./results/iterations_results_scipy.npy",allow_pickle = True)
    fig=plt.figure()
    arr_n_plus_m = [_tuple[0]+_tuple[1] for _tuple in largest_coeff_float]
    arr_min_n_m = [min(_tuple[0], _tuple[1]) for _tuple in largest_coeff_float]
    largest_coeff_float_time = [_tuple[2] for _tuple in largest_coeff_float]
    largest_coeff_fraction_time = [_tuple[2] for _tuple in largest_coeff_fraction]
    largest_increase_float_time = [_tuple[2] for _tuple in largest_increase_float]
    largest_increase_fraction_time = [_tuple[2] for _tuple in largest_increase_fraction]
    scipy_time = [_tuple[2] for _tuple in scipy_res]
    plt.scatter(arr_n_plus_m, largest_coeff_float_time, color='r', marker = ".", label ="largest_coeff_float")
    plt.scatter(arr_n_plus_m, largest_coeff_fraction_time, color='r', marker = "*", label ="largest_coeff_fraction")
    plt.scatter(arr_n_plus_m, largest_increase_float_time, color='b', marker = ".", label ="largest_increase_float")
    plt.scatter(arr_n_plus_m, largest_increase_fraction_time, color='b', marker = "*", label ="largest_increase_fraction")
    plt.scatter(arr_n_plus_m, scipy_time, color='g', marker = ".", label ="scipy")

    # plt.scatter(arr_min_n_m, largest_coeff_float_time, color='r', marker = ".", label ="largest_coeff_float")
    # plt.scatter(arr_min_n_m, largest_coeff_fraction_time, color='r', marker = "o", label ="largest_coeff_fraction")
    # plt.scatter(arr_min_n_m, largest_increase_float_time, color='b', marker = ".", label ="largest_increase_float")
    # plt.scatter(arr_min_n_m, largest_increase_fraction_time, color='b', marker = "o", label ="largest_increase_fraction")
    # plt.scatter(arr_min_n_m, scipy_time, color='g', marker = "o", label ="scipy")

    plt.scatter
    # plt.xlabel('n+m')
    plt.xlabel('min(n,m)')
    plt.ylabel('Time')
    plt.title('Time experiment for min(m,n)')
    plt.legend()
    plt.show()

def print_integer_experiment():
    res_integer = np.load("./results/integer/iterations_results_largest_increase_integer.npy", allow_pickle=True)
    res_fraction = np.load("./results/integer/iterations_results_largest_increase_fraction.npy", allow_pickle=True)

    fig=plt.figure()
    arr_n_plus_m = [_tuple[0]+_tuple[1] for _tuple in res_integer]
    arr_min_n_m = [min(_tuple[0], _tuple[1]) for _tuple in res_fraction]
    integer_time = [_tuple[2] for _tuple in res_integer]
    fraction_time = [_tuple[2] for _tuple in res_fraction]

    plt.scatter(arr_n_plus_m, integer_time, color='b', marker='.', label="integer")
    plt.scatter(arr_n_plus_m, fraction_time, color='r', marker='.', label="fraction")

    # plt.scatter(arr_min_n_m, integer_time, color='b', marker='.', label="float")
    # plt.scatter(arr_min_n_m, fraction_time, color='r', marker='.', label="fraction")

    plt.scatter
    plt.xlabel('n+m')
    # plt.xlabel('min(n,m)')
    plt.ylabel('Time')
    plt.title('Time experiment for n+m')
    plt.legend()
    plt.show()

def run_and_print_experiments():
    run_experiment_1(100)
    run_experiment_2(200)
    run_experiment_inter_vs_fraction_pivot(200)
    print_experiment_1()
    print_experiment_2()
    print_integer_experiment()

#run_examples()
#test_correctness_of_algorithms(20)
run_and_print_experiments()

