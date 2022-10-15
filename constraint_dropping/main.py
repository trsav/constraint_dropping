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
    maximize,
)
import time
from prettytable import PrettyTable
from pyomo.opt import SolverFactory
from utils import plot_result
from utils import create_lp, names_to_list
import multiprocessing as mp

# logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# Extension of Boyd
# Extension of Lubin

# https://web.stanford.edu/~boyd/papers/pdf/prac_robust.pdf
# https://link.springer.com/content/pdf/10.1007/s10287-015-0236-z.pdf


# TODO
# Geometric Mean
# Reformulation
# Ellipsoid
# Instance space analysis
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
        # if os.path.getsize(path) > 1000000:
        #     print("File too large for this analysis \n")
        #     continue

        try:
            (
                lp,
                A,
                b,
                c,
                x,
                p,
                eq_con_list,
                ineq_con_list,
                obj,
                ineq_dict,
                eq_dict,
                cn,
                rn,
            ) = create_lp(path)
        except ValueError:
            print("Error in MPS parsing \n")
            continue
        except IndexError:
            print("Error in MPS parsing \n")
            continue

        if len(ineq_con_list) == 0:
            print(case + " has no inequality constraints... ")
            continue

        def var_bounds(m, i):
            return (x[i][0], x[i][1])

        s_parse = time.time()
        epsilon = 1e-8
        m_upper = ConcreteModel()
        m_upper.x = Set(initialize=x.keys())
        m_upper.x_v = Var(m_upper.x, bounds=var_bounds)
        m_upper.cons = ConstraintList()

        for i in range(len(ineq_con_list)):
            con = ineq_con_list[i]
            ni = ineq_dict[i]
            p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
            p_keys = [pi for pi in p_keys if p[pi]["val"] != 0]
            p_nominal = {}
            for j in list(p_keys):
                p_nominal[j] = p[j]["val"]
            m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) <= 0)

        for i in range(len(eq_con_list)):
            con = eq_con_list[i]
            ni = eq_dict[i]
            p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
            p_keys = [pi for pi in p_keys if p[pi]["val"] != 0]
            p_nominal = {}
            for j in list(p_keys):
                p_nominal[j] = p[j]["val"]
            m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) == 0)

        m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)

        res = SolverFactory("gurobi").solve(m_upper)
        term_con = res.solver.termination_condition
        if term_con is TerminationCondition.infeasible:
            print("Problem is nominally infeasible...")
            continue

        if term_con is TerminationCondition.infeasibleOrUnbounded:
            print("Problem is nominally infeasible...")
            continue
        t_lp = time.time() - s_parse
        x_opt = {}

        for x_name, x_data in m_upper.x_v._data.items():
            x_opt[x_name] = x_data.value

        global solve_subproblem

        def solve_subproblem(j, x_opt, warm):
            con = ineq_con_list[j]
            ni = ineq_dict[j]
            p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
            non_zero_index = [i for i in range(len(p_keys)) if p[p_keys[i]]["val"] != 0]
            p_keys = [p_keys[i] for i in non_zero_index]
            var_keys = [
                (pi.split("v_")[0]) + "v"
                for pi in p_keys
                if "c" not in (pi.split("v_")[0]) + "v"
            ]
            x_opt_new = {}
            for var in var_keys:
                x_opt_new[var] = x_opt[var]
            x_opt = x_opt_new

            def uncertain_bounds(m, i):
                return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])

            m_lower = ConcreteModel()
            m_lower.p = Set(initialize=p_keys)
            m_lower.p_v = Var(m_lower.p, within=Reals, bounds=uncertain_bounds)
            m_lower.obj = Objective(expr=con(x_opt, m_lower.p_v), sense=maximize)

            for k in p_keys:
                m_lower.p_v[k].set_value(warm[k])

            SolverFactory("gurobi").solve(m_lower)
            p_opt = {}
            for p_name, p_data in m_lower.p_v._data.items():
                p_opt[p_name] = p_data.value
            obj_v = value(m_lower.obj)
            if obj_v < epsilon:
                return [p_opt, obj_v, True]
            else:
                return [p_opt, obj_v, False]

        p_warm = []
        for i in range(len(ineq_con_list)):
            p_start = {}
            ni = ineq_dict[i]
            p_keys = [cn[k] + "_" + rn[ni] for k in range(len(cn))] + [rn[ni]]
            p_keys = [pi for pi in p_keys if p[pi]["val"] != 0]
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
                pool.close()
            else:
                res = []
                for j in range(len(ineq_con_list)):
                    res.append(solve_subproblem(j, x_opt, p_warm[j]))

            n_cv = 0
            m_cv = 0

            for i in range(len(res)):
                p_opt, cv, flag = res[i]
                if flag is False:
                    con = ineq_con_list[i]
                    m_upper.cons.add(expr=con(m_upper.x_v, p_opt) <= 0)
                    n_cv += 1
                if cv > m_cv:
                    m_cv = cv
                p_warm[i] = p_opt

            t_it = time.time() - s_parse

            info.add_row(
                [
                    iteration,
                    n_cv,
                    (n_cv / n_c) * 100,
                    m_cv,
                    len(m_upper.cons),
                    t_it,
                    t_it / t_lp,
                ]
            )
            print(info[-1])
            iteration += 1

            if n_cv == 0:
                print(info)
                print("Robust Objective: ", value(m_upper.obj))
                info.title = ""
                with open("outputs/" + case + "_standard.csv", "w") as f:
                    f.write(info.get_csv_string())
                break

            res = SolverFactory("gurobi").solve(m_upper)
            term_con = res.solver.termination_condition
            if term_con is TerminationCondition.infeasible:
                print("Problem is robustly infeasible...")
                break

            if term_con is TerminationCondition.infeasibleOrUnbounded:
                print("Problem is robustly infeasible...")
                break

            x_opt = {}
            for x_name, x_data in m_upper.x_v._data.items():
                x_opt[x_name] = x_data.value


run_all()
plot_result()
