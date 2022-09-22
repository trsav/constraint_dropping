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
from pyomo.opt import SolverFactory
import logging
from utils import create_lp, parse_lp
import multiprocessing as mp

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# TODO
# Add warm starting
# Extension of Boyd
# ML based on features
# Cutting planes


case = "israel"

info_dict = {}

path = "lp_files/expanded_lp/" + case + ".mps"
x, p, eq_con_list, ineq_con_list, obj, ineq_dict, eq_dict, cn, rn = create_lp(path)
lp = parse_lp(path)


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
    m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) <= 0)


for i in range(len(eq_con_list)):
    con = eq_con_list[i]
    ni = eq_dict[i]
    p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
    p_nominal = {}
    for i in list(p_keys):
        p_nominal[i] = p[i]["val"]
    m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) == 0)

m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
SolverFactory("gurobi").solve(m_upper)

x_opt = {}
for x_name, x_data in m_upper.x_v._data.items():
    x_opt[x_name] = x_data.value


def solve_subproblem(j, x_opt):

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
        m_lower.p_v[k].set_value(p[k]["val"])
        if p[k]["unc"] == 0:
            m_lower.p_v[k].fix(p[k]["val"])

    SolverFactory("gurobi").solve(m_lower)

    if value(m_lower.obj) < epsilon:
        return "Feasible"

    else:

        p_opt = {}
        for p_name, p_data in m_lower.p_v._data.items():
            if p_data.value is None:
                p_opt[p_name] = p[p_name]["val"]
            else:
                p_opt[p_name] = p_data.value

        return [p_opt, value(m_lower.obj)]


pool = mp.Pool(mp.cpu_count() - 1)
while True:

    # res = []
    # for j in range(len(ineq_con_list)):
    #     res.append(solve_subproblem(j,x_opt))

    res = pool.starmap(
        solve_subproblem, [(i, x_opt) for i in range(len(ineq_con_list))]
    )

    n_cv = 0
    m_cv = 0
    for i in range(len(res)):
        p_res = res[i]
        if p_res != "Feasible":
            p_opt = p_res[0]
            cv = p_res[1]
            con = ineq_con_list[i]
            m_upper.cons.add(expr=con(m_upper.x_v, p_opt) <= 0)
            n_cv += 1
            if cv > m_cv:
                m_cv = cv

    print("Constraint Violations\t", n_cv)
    print("Maximum Constraint Violation\t", m_cv)
    if m_cv == 0:
        break

    SolverFactory("gurobi").solve(m_upper)

    x_opt = value(m_upper.x_v[:])
    x_opt = {}
    for x_name, x_data in m_upper.x_v._data.items():
        x_opt[x_name] = x_data.value
