from pyomo.environ import (
    ConcreteModel,
    Var,
    TerminationCondition,
    value,
    Set,
    ConstraintList,
    Objective,
    minimize,
    Reals,
    VarList,
    sqrt,
)
import numpy as np
import os
from prettytable import PrettyTable
from pyomo.opt import SolverFactory
from utils import parse_lp, parse_variables, parse_matrices


# logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# Extension of Boyd
# Extension of Lubin

# https://web.stanford.edu/~boyd/papers/pdf/prac_robust.pdf
# https://link.springer.com/content/pdf/10.1007/s10287-015-0236-z.pdf


# TODO
# Instance space analysis
# Geometric Mean
# Reformulation
# Ellipsoid
# ML based on features
# RL?


def run_all_reformulated(*args):
    names = args[0]
    type = args[1]
    if type == "Ellipse":
        prob = args[2]
        g = np.sqrt((-2 * np.log(prob)))
    for case in names:

        info = PrettyTable()
        info.title = case
        info.field_names = [
            "Iteration",
            "Robust Constraint Violations",
            "Robust Constraint Violations (%)",
            "Maximum Constraint Violation",
            "Upper Level Problem Size",
            "Time (s)",
            "Nominal Time Factor",
        ]

        path = "lp_files/expanded_lp/" + case + ".mps"
        if os.path.getsize(path) > 1000000:
            print("File too large for this analysis \n")
            continue

        def var_bounds(m, i):
            return (x[i][0], x[i][1])

        unc_percen = 5 / 100  # percentage uncertainty
        m = ConcreteModel()
        lp = parse_lp(path)
        x = parse_variables(lp)
        cn = lp["col_names"]
        rn = lp["row_names"]
        A, b, c = parse_matrices(lp)
        types = lp["types"]

        m.x = Set(initialize=x.keys())
        m.x_v = Var(m.x, bounds=var_bounds)
        m.cons = ConstraintList()

        def linear_eq(x, a, b):
            s = 0
            for j in range(len(a)):
                if a[j] != 0:
                    s += a[j] * x[cn[j]]
            return s - b

        m.u = VarList(domain=Reals)
        for i in range(len(rn)):
            if types[i] != "E":
                if type == "Box":
                    s_in = []
                    s = 0
                    for j in range(len(A[i, :])):
                        if A[i, j] != 0.0:
                            s_in.append(j)
                            xv = m.x_v[cn[j]]
                            u = m.u.add()
                            s += u
                            m.cons.add(expr=-u <= (A[i, j] * unc_percen * xv))
                            # print(m.cons[len(m.cons)].expr)
                            m.cons.add(expr=(A[i, j] * unc_percen * xv) <= u)
                            # print(m.cons[len(m.cons)].expr)
                    sn = 0
                    xvals = [m.x_v[cn[k]] for k in s_in]
                    avals = [A[i, k] for k in s_in]
                    for k in range(len(xvals)):
                        sn += xvals[k] * avals[k]
                    # print(sn+s-b[i])
                    m.cons.add(expr=sn + s - b[i] <= 0)

                if type == "Ellipse":
                    soc = 0
                    s = 0
                    for j in range(len(A[i, :])):
                        if A[i, j] != 0.0:
                            soc += (A[i, j] * unc_percen * m.x_v[cn[j]]) ** 2
                            s += A[i, j] * m.x_v[cn[j]]

                    lhs = g * sqrt(soc)
                    rhs = b[i] - s
                    print("\n")
                    print(lhs)
                    print("\n")
                    # (x,r) where,
                    # (x1**2 +...xn**2 <= r1**2 +...+rn**2)
                    print(rhs)
                    print("\n")
                    # m.cons.add(expr= lhs <= rhs)
                    # m.cons.add(expr= kernel.conic.quadratic(b[i]-s,g**2)

            if types[i] == "E":
                m.cons.add(expr=linear_eq(m.x_v, A[i, :], b[i]) == 0)

        def obj(x):
            s = 0
            for i in range(len(cn)):
                s += c[i] * x[cn[i]]
            return s

        m.obj = Objective(expr=obj(m.x_v), sense=minimize)

        res = SolverFactory("gurobi").solve(m)
        term_con = res.solver.termination_condition
        if term_con is TerminationCondition.infeasible:
            print("Problem is nominally infeasible...")
            continue

        if term_con is TerminationCondition.infeasibleOrUnbounded:
            print("Problem is nominally infeasible...")
            continue

        print("Robust Objective via reformulation: ", value(m.obj))
        # t_lp = time.time() - s_parse
        # x_opt = {}

        # for x_name, x_data in m_upper.x_v._data.items():
        #     x_opt[x_name] = x_data.value

        #     t_it = time.time() - s_parse
