import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


"""
# Notes
1. discount (gamma) =0.95
2. policy: probability of moving in any of the four directions = 0.25
3. Any action at blue => +5 reward => jump to the red square
4. Any action at green => +2.5 reward => jump to the red or yellow square with probability 0.5

# Objectives
(section 1): Estimate the value function for each of the states
1. Solve the system of Bellman equations explicitly             => how can i explicitly solve the equations?
2. Iterative policy evaluation                                  => what is iterative policy evaluation?
3. value iteration.                                             => what is value iteration?
Which states have the highest value? Does this surprise you?

(section 2):  Determine the optimal policy for the gridworld problem
1. explicitly solving the Bellman optimality equation
2. using policy iteration with iterative policy evaluation 
3. policy improvement with value iteration.
"""


class Agent:
    def __init__(self):
        self.discount = 0.95
        self.policy_probability = 0.25
        self.expected_rewards = np.zeros((5, 5))

    def explicit(self, g=0.95):
        """
        (i,j) in (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
        v_ij = max [P(s_ij+1 | s_ij, a1) * discount * v_ij+1 +
                        P(s_ij-1 | s_ij, a2) * discount * v_ij-1 +
                        P(s_i-1j | s_ij, a3) * discount * v_i-1j +
                        P(s_i+1j | s_ij, a4) * discount * v_i+1j]
        given the probabilities are equal we have:
        v_ij = max[0.25 * discount * (v_ij+1 + v_ij-1 + v_i-1j + v_i+1j)]
        ___________________________________________________________________________________________
        eq1 = 0.25*g*(v12 + v10 + v01 + v21) - v11
        eq2 = 0.25*g*(v13 + v11 + v02 + v22) - v12
        eq3 = 0.25*g*(v14 + v12 + v03 + v23) - v13

        eq4 = 0.25*g*(v22 + v20 + v11 + v31) - v21
        eq5 = 0.25*g*(v23 + v21 + v12 + v32) - v22
        eq6 = 0.25*g*(v24 + v22 + v13 + v33) - v23

        eq7 = 0.25*g*(v32 + v30 + v21 + v41) - v31
        eq8 = 0.25*g*(v33 + v31 + v22 + v42) - v32
        eq9 = 0.25*g*(v34 + v32 + v23 + v43) - v33
        ___________________________________________________________________________________________

        (i,j) in (4, 1), (4, 2), (4, 3)
        v_ij = max[0.25 * discount * (-1/discount + v_ij + v_i-1j + v_ij-1 + v_ij+1)]
        ___________________________________________________________________________________________
        eq10 = 0.25*g*(-1/g + v41 + v31 + v40 + v42) - v41
        eq11 = 0.25*g*(-1/g + v42 + v32 + v41 + v43) - v42
        eq12 = 0.25*g*(-1/g + v43 + v33 + v42 + v44) - v43

        ___________________________________________________________________________________________
        (i,j) in (1, 0), (2, 0), (3, 0)
        v_ij = max[0.25 * discount * (-1/discount + v_ij + v_i+1j + v_i-1j + v_ij+1)]
        ___________________________________________________________________________________________
        eq13 = 0.25*g*(-1/g + v10 + v20 + v00 + v11) - v10
        eq14 = 0.25*g*(-1/g + v20 + v30 + v10 + v21) - v20
        eq15 = 0.25*g*(-1/g + v30 + v40 + v20 + v31) - v30

        ___________________________________________________________________________________________
        (i,j) in (1, 4), (2, 4), (3, 4)
        v_ij = max[0.25 * discount * (-1/discount + v_ij + v_i-1j + v_ij-1 + v_i+1j)]
        ___________________________________________________________________________________________
        eq16 = 0.25*g*(-1/g + v14 + v04 + v13 + v24) - v14
        eq17 = 0.25*g*(-1/g + v24 + v14 + v23 + v34) - v24
        eq18 = 0.25*g*(-1/g + v34 + v24 + v33 + v44) - v34

        ___________________________________________________________________________________________
        (i,j) in (0, 2), (0, 3)
        v_ij = max[0.25 * discount * (-1/discount + v_ij + v_ij+1 + v_i+1j + v_ij-1)]
        ___________________________________________________________________________________________
        eq19 = 0.25*g*(-1/g + v02 + v03 + v12 + v01) - v02
        eq20 = 0.25*g*(-1/g + v03 + v04 + v13 + v02) - v03

        ___________________________________________________________________________________________
        (0, 1) +5!
        eq21 = ?

        ___________________________________________________________________________________________
        (0, 0)
        v_ij = max[0.25 * discount * (-2/discount + 2*v_ij + v_ij+1 + v_i+1j)]
        eq22 = 0.25*g*(-2/g + 2*v00 + v01 + v10) - v00

        ___________________________________________________________________________________________
        (0, 4) + 2.5!
        v_ij = max[0.25 * discount * (-2/discount + 2*v_ij + v_ij-1 + v_i-1j)]
        eq23 = ?

        ___________________________________________________________________________________________
        (4, 0)
        v_ij = max[0.25 * discount * (-2/discount + 2*v_ij + v_ij+1 + v_i-1j)]
        eq24 = 0.25*g*(-2/g + 2*v40 + v41 + v30) - v40

        ___________________________________________________________________________________________
        (4, 4)
        v_ij = max[0.25 * discount * (-2/discount + 2*v_ij + v_ij-1 + v_i-1j)]
        eq25 = 0.25*g*(-2/g + 2*v44 + v43 + v34 ) - v44
        """
        def equations(p):
            v00, v01, v02, v03, v04, v10, v11, v12, v13, v14, v20, v21, v22, v23, v24, v30, v31, v32, v33, v34, v40, v41, v42, v43, v44 = p
            eq1 = 0.25 * g * (v12 + v10 + v01 + v21) - v11
            eq2 = 0.25 * g * (v13 + v11 + v02 + v22) - v12
            eq3 = 0.25 * g * (v14 + v12 + v03 + v23) - v13

            eq4 = 0.25 * g * (v22 + v20 + v11 + v31) - v21
            eq5 = 0.25 * g * (v23 + v21 + v12 + v32) - v22
            eq6 = 0.25 * g * (v24 + v22 + v13 + v33) - v23

            eq7 = 0.25 * g * (v32 + v30 + v21 + v41) - v31
            eq8 = 0.25 * g * (v33 + v31 + v22 + v42) - v32
            eq9 = 0.25 * g * (v34 + v32 + v23 + v43) - v33

            eq10 = 0.25 * g * (-1 / g + v41 + v31 + v40 + v42) - v41
            eq11 = 0.25 * g * (-1 / g + v42 + v32 + v41 + v43) - v42
            eq12 = 0.25 * g * (-1 / g + v43 + v33 + v42 + v44) - v43


            return
        pass


if __name__ == '__main__':
    pass
