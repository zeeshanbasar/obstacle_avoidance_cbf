import numpy as np
import casadi as ca

from src import limits
from src.vehicle import f
from src.obstacle import Obstacle

class Solver():

    def __init__(self, x_dim, u_dim, obs=[], dt=0.02, n_obs=1, max_n_obs=1):

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.obs = obs # each obstacle instance is created outside and sent as a whole package

        self.dt = dt

        self.W = ca.DM([1/limits.A_MAX**2, 1/ca.tan(limits.PHI_MAX)**2])

        self.slack_weight = 1e4
        self.slack_tolerance = 1e-6

        self.t_phi_prev = 0.0
        self.d = limits.PHI_DOT_MAX * self.dt

        self.n_obs = len(obs)

        self.max_n_obs = max_n_obs

        self.A_pad = np.zeros((self.max_n_obs - self.n_obs, self.u_dim))
        self.b_pad = np.zeros((self.max_n_obs - self.n_obs, 1))

        self.setup_solver()

    def setup_solver(self):

        nx, nu = self.x_dim, self.u_dim

        opti = ca.Opti('conic')

        u_var = opti.variable(nu, 1)

        current_x_var = opti.parameter(nx, 1)
        uncert_u_nom = opti.parameter(nu, 1)

        slack_var = opti.variable(self.max_n_obs, 1)

        A = opti.parameter(self.max_n_obs,2)
        b = opti.parameter(self.max_n_obs,1)

        t_hi = opti.parameter()
        t_lo = opti.parameter()

        # CBF constraint
        opti.subject_to(A@u_var >= b - slack_var)

        # Slack constraint
        opti.subject_to(slack_var >= 0.0)

        cost = 0.5 * ca.dot((uncert_u_nom - u_var), self.W * (uncert_u_nom - u_var)) \
            +  self.slack_weight * ca.dot(slack_var, slack_var)

        # Input constraint
        opti.subject_to(u_var[0] <= limits.A_MAX)
        opti.subject_to(u_var[0] >= -limits.A_MAX)
        opti.subject_to(u_var[1] <= ca.tan(limits.PHI_MAX))
        opti.subject_to(u_var[1] >= ca.tan(-limits.PHI_MAX))

        # Roll rate constraint
        opti.subject_to(u_var[1] <= t_hi)
        opti.subject_to(u_var[1] >= t_lo)

        # Minimize
        opti.minimize(cost)

        # Set verbosity option of optimizer
        opts = {'printLevel': 'low', 'error_on_fail': False}

        # Select QP solver
        opti.solver('qpoases', opts)

        self.opti_dict = {
            'opti': opti,
            'u_var': u_var,
            'slack_var': slack_var,
            'current_x_var': current_x_var,
            'uncert_u_nom': uncert_u_nom,
            'A':A,
            'b':b,
            't_hi':t_hi,
            't_lo':t_lo,
            'cost': cost            
        }

    def solve(self, x, u_nom):

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        u_var = opti_dict['u_var']
        slack_var = opti_dict['slack_var']
        current_state_var = opti_dict['current_x_var']
        uncertified_action_var = opti_dict['uncert_u_nom']
        A, b = opti_dict['A'], opti_dict['b']
        t_hi, t_lo = opti_dict['t_hi'], opti_dict['t_lo']
        cost = opti_dict['cost']

        opti.set_value(current_state_var, x)
        opti.set_value(uncertified_action_var, u_nom)

        # CBF constraint - set value
        _A, _b = np.array([]), np.array([])
        for obs in self.obs:
            A_temp, b_temp = obs.constraint(x=x,
                                    u=u_nom)
            
            _A = np.vstack((_A, A_temp)) if _A.size else A_temp
            _b = np.vstack((_b, b_temp)) if _b.size else b_temp

        _A = np.vstack((_A, self.A_pad)) if _A.size else self.A_pad
        _b = np.vstack((_b, self.b_pad)) if _b.size else self.b_pad
        
        opti.set_value(A, _A)
        opti.set_value(b, _b)

        # Roll rate constraint - set value
        phi_prev = np.arctan(self.t_phi_prev)
        self.d = limits.PHI_DOT_MAX * self.dt
        opti.set_value(t_lo, ca.tan(max(phi_prev - self.d, -limits.PHI_MAX)))
        opti.set_value(t_hi, ca.tan(min(phi_prev + self.d, limits.PHI_MAX)))

        try:
            # Solve optimization problem
            sol = opti.solve()
            feasible = True

            certified_action = sol.value(u_var)
            self.t_phi_prev = certified_action[1]

            slack_val = sol.value(slack_var)
            s_real = slack_val[:self.n_obs] if self.n_obs > 1 else slack_val
            if np.max(s_real, initial=0.0) > self.slack_tolerance:
                print('\nFailed: Slack greater than tolerance')
                print('Slack:', slack_val[:self.n_obs]) if self.n_obs > 1 else print('Slack:', slack_val)
                print('------------------------------------------------')
                feasible = False
            c = sol.value(cost)
        except RuntimeError as e:
            print(e)
            feasible = False
            certified_action = opti.debug.value(u_var)
            print('Certified_action:', certified_action)
            slack_val = opti.debug.value(slack_var)
            print('Slack:', slack_val)
            # print('Lie Derivative:', self.lie_derivative(X=current_state, u=certified_action)['LfV'])
            # print('Linear Function:', self.linear_func(x=self.cbf(X=current_state)['cbf'])['y'])
            print('------------------------------------------------')
        return certified_action, slack_val

    def reset(self, tphi0=0.0):

        self.t_phi_prev = tphi0

    def set_obstacles(self, obstacles):


        self.obs = obstacles
        self.n_obs = len(obstacles)
        self.A_pad = np.zeros((self.max_n_obs - self.n_obs, self.u_dim))
        self.b_pad = np.zeros((self.max_n_obs - self.n_obs, 1))