from pyomo.environ import (
    ConcreteModel,
    Var,
    Reals,
    value,
    Set,
    ConstraintList,
    Objective,
    minimize,
    maximize,
)

from prettytable import PrettyTable
from pyomo.opt import SolverFactory
import logging
from utils import create_lp, parse_lp, parse_matrices, names_to_list
import multiprocessing as mp

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# TODO
# Extension of Boyd
# ML based on features
# Cutting planes

names = names_to_list()
names = ["adlittle"]

for case in names:

    print("Starting to solve", case)
    info = PrettyTable()
    info.title = case
    info.field_names = [
        "Iteration",
        "Number of Robust Constraint Violations",
        "Percentage of Robust Constraint Violations",
        "Maximum Constraint Violation",
    ]

    path = "lp_files/expanded_lp/" + case + ".mps"
    x, p, eq_con_list, ineq_con_list, obj, ineq_dict, eq_dict, cn, rn = create_lp(path)
    if len(ineq_con_list) == 0:
        print(case + " has no inequality constraints... ")
        continue
    lp = parse_lp(path)
    (
        A,
        b,
        c,
    ) = parse_matrices(lp)

    def var_bounds(m, i):
        return (x[i][0], x[i][1])

    epsilon = 1e-4
    m_upper = ConcreteModel()
    m_upper.x = Set(initialize=x.keys())
    m_upper.x_v = Var(m_upper.x, bounds=var_bounds)
    m_upper.cons = ConstraintList()

    for i in range(len(ineq_con_list)):
        con = ineq_con_list[i]
        ni = ineq_dict[i]
        p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
        p_nominal = {}
        for i in list(p_keys):
            p_nominal[i] = p[i]["val"]
        if min(p_nominal.values()) == 0 and max(p_nominal.values()) == 0:
            pass
        else:
            m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) <= 0)

    for i in range(len(eq_con_list)):
        con = eq_con_list[i]
        ni = eq_dict[i]
        p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
        p_nominal = {}
        for i in list(p_keys):
            p_nominal[i] = p[i]["val"]
        if min(p_nominal.values()) == 0 and max(p_nominal.values()) == 0:
            pass
        else:
            m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) == 0)

    m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
    SolverFactory("gurobi").solve(m_upper)

    x_opt = {}
    for x_name, x_data in m_upper.x_v._data.items():
        x_opt[x_name] = x_data.value

    def solve_subproblem(j, x_opt, warm):

        con = ineq_con_list[j]
        ni = ineq_dict[j]
        p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]

        def uncertain_bounds(m, i):
            return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])

        m_lower = ConcreteModel()

        m_lower.p = Set(initialize=p_keys)
        m_lower.p_v = Var(m_lower.p, within=Reals, bounds=uncertain_bounds)
        m_lower.obj = Objective(expr=con(x_opt, m_lower.p_v), sense=maximize)

        for k in p_keys:
            m_lower.p_v[k].set_value(warm[k])
            if p[k]["unc"] == 0:
                m_lower.p_v[k].fix(p[k]["val"])

        SolverFactory("gurobi").solve(m_lower)

        p_opt = {}
        for p_name, p_data in m_lower.p_v._data.items():
            if p_data.value is None:
                p_opt[p_name] = p[p_name]["val"]
            else:
                p_opt[p_name] = p_data.value

        if value(m_lower.obj) < epsilon:
            return [p_opt]
        else:
            return [p_opt, value(m_lower.obj)]

    p_warm = []
    for i in range(len(ineq_con_list)):
        p_start = {}
        ni = ineq_dict[i]
        p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
        for k in p_keys:
            p_start[k] = p[k]["val"]
        p_warm.append(p_start)

    n_c = len(ineq_con_list)

    parallel = True

    iteration = 1

    while True:
        if parallel is True:
            pool = mp.Pool(mp.cpu_count() - 2)
            res = pool.starmap(
                solve_subproblem,
                [(i, x_opt, p_warm[i]) for i in range(len(ineq_con_list))],
            )
        else:
            res = []
            for j in range(len(ineq_con_list)):
                res.append(solve_subproblem(j, x_opt, p_warm[j]))

        n_cv = 0
        m_cv = 0

        for i in range(len(res)):
            p_res = res[i]
            if len(p_res) > 1:
                p_opt = p_res[0]
                cv = p_res[1]
                con = ineq_con_list[i]
                m_upper.cons.add(expr=con(m_upper.x_v, p_opt) <= 0)
                n_cv += 1
                if cv > m_cv:
                    m_cv = cv
            else:
                p_opt = p_res[0]

            p_warm[i] = p_opt

        info.add_row([iteration, n_cv, (n_cv / n_c) * 100, m_cv])

        iteration += 1

        print(info, end="\n")
        if m_cv == 0:
            break

        SolverFactory("gurobi").solve(m_upper)

        x_opt = {}
        for x_name, x_data in m_upper.x_v._data.items():
            x_opt[x_name] = x_data.value
