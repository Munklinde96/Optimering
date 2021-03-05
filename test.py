import unittest
import numpy as np
from fractions import Fraction

from simplex import get_dual_dictionary, lp_solve, Dictionary, bland, LPResult, exercise2_5, phase1_alg

class TestExample1(unittest.TestCase):
    def setUp(self):
        self.c = np.array([5,4,3])
        self.A = np.array([[2,3,1],[4,1,2],[3,4,2]])
        self.b = np.array([5,11,8])

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        for (a,b) in zip(list(D.basic_solution()), [2.0, 0.0, 1.0]) :
            self.assertAlmostEqual(a,b)
            
    def test_infeasable_primal_unbounded_dual(self):
        print("infeasible test of ex. 2.5")
        c,A,b = exercise2_5() # unbounded
        d = Dictionary(c,A,b)
        dual = get_dual_dictionary(d)  #dual of undbounded is infeasable
        res, D = phase1_alg(dual, bland(dual, 1e-5))
        assert(res == LPResult.UNBOUNDED)
        assert(D == None)


if __name__ == '__main__':
    unittest.main()
