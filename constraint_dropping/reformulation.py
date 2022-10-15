from pyomo.environ import (
    ConcreteModel,
    Var,
    TerminationCondition,
    Reals,
    value,
    Set,
    ConstraintList,
    Objective,
    minimize,
    VarList,
)
import os
from prettytable import PrettyTable
from pyomo.opt import SolverFactory
from utils import plot_result
from utils import parse_lp, parse_variables, parse_matrices
from utils import names_to_list


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


def run_all():

    names = names_to_list()
    names = ["brandy"]
    for case in names:

        print("Starting to solve", case)
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

        def linear_con(x, a, b, t):
            s = 0
            for i in range(len(a)):
                if a[i] != 0:
                    s += a[i] * x[cn[i]]
                return s - b

        def linear_eq(x, a, b):
            s = 0
            for j in range(len(a)):
                if a[j] != 0:
                    s += a[j] * x[cn[j]]
            return s - b

        m.u = VarList(domain=Reals)
        for i in range(len(rn)):
            if types[i] != "E":
                s_in = []
                s = 0
                for j in range(len(A[i, :])):
                    if A[i, j] != 0.0:
                        s_in.append(j)
                        xv = m.x_v[cn[j]]
                        u = m.u.add()
                        s += u
                        m.cons.add(expr=-u <= (A[i, j] * unc_percen * xv))
                        m.cons.add(expr=(A[i, j] * unc_percen * xv) <= u)
                sn = 0
                xvals = [m.x_v[cn[k]] for k in s_in]
                avals = [A[i, k] for k in s_in]
                for k in range(len(s_in)):
                    sn += xvals[k] * avals[k]
                m.cons.add(expr=sn + s - b[i] <= 0)

                # m.cons.add(expr=linear_con(m.x_v,A[i,:],b[i],types[i]) <= 0)
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

        print(value(m.obj))
        # t_lp = time.time() - s_parse
        # x_opt = {}

        # for x_name, x_data in m_upper.x_v._data.items():
        #     x_opt[x_name] = x_data.value

        #     t_it = time.time() - s_parse


plot_result()
run_all()
