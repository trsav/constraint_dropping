import time
import numpy as np
from pysmps import smps_loader as smps
import logging
from cmath import inf

logging.getLogger("pyomo.core").setLevel(logging.ERROR)


def parse_lp(path):
    """
    name:               The name given to the linear program (can't be blank)
    objective_name:     The name of the objective function value
    row_names:          list of row names
    col_names:          list of column names
    types:              list of constraint type indicators,
                        i.e. either "E", "L" or "G" for equality,
                        lower/equal or greater/equal constraint respectively.
    c:                  the objective function coefficients
    A:                  the constraint matrix
    rhs_names:          list of names of right hand sides (there can be multiple
                        right hand side components be defined,
                        seldom more than one though)
    rhs:                dictionary (rhs_name) => b, where b is the vector
                        of constraint values for that given right hand side name.
    bnd_names:          list of names of box-bounds (seldom more than one)
    bnd:                dictionary (bnd_name) => {"LO": v_l, "UP": v_u}
                        where v_l is the vector of lower bounds and v_u is the vector of
                        upper bound values (defaults to v_l = 0 and v_u = +inf).

    Of the form:

    min 	c * x

    s.t.	for each rhs_name with corresponding b:

                A[types == "E",:] * x  = b[types == "E"]
                A[types == "L",:] * x <= b[types == "L"]
                A[types == "G",:] * x >= b[types == "G"]

            for each bnd_name with corresponding v_l and v_u:

                v_l <= x < v_u

    """
    lp_dict = {}
    lp = smps.load_mps(path)
    fields = [
        "name",
        "objective name",
        "row_names",
        "col_names",
        "col_types",
        "types",
        "c",
        "A",
        "rhs_names",
        "rhs",
        "bnd_names",
        "bnd",
    ]
    for i in range(len(fields)):
        lp_dict[fields[i]] = lp[i]

    for i in range(len(lp_dict["col_names"])):
        lp_dict["col_names"][i] += "_v"
    for i in range(len(lp_dict["row_names"])):
        lp_dict["row_names"][i] += "_c"

    return lp_dict


def parse_matrices(lp):
    A = lp["A"]
    if len(lp["rhs"]) == 0:
        b = np.zeros(len(A[:, 0]))
    else:
        b = lp["rhs"][list(lp["rhs"].keys())[0]]
    c = lp["c"]
    return A, b, c


def parse_parameters(lp, percentage_uncertainty):
    pu = percentage_uncertainty
    A, b, c = parse_matrices(lp)
    n = len(c)
    m = len(b)
    p = {}
    for i in range(n):
        p[lp["col_names"][i]] = {"val": c[i], "unc": c[i] * pu / 100}
        for j in range(m):
            p[lp["col_names"][i] + "_" + lp["row_names"][j]] = {
                "val": A[j, i],
                "unc": A[j, i] * pu / 100,
            }
    for i in range(m):
        p[lp["row_names"][i]] = {"val": b[i], "unc": b[i] * pu / 100}
    return p


def parse_variables(lp):
    x = {}
    if len(lp["bnd"]) == 0:
        for i in range(len(lp["col_names"])):
            x[lp["col_names"][i]] = [0, inf]
    else:
        lb = list(lp["bnd"].values())[0]["LO"]
        ub = list(lp["bnd"].values())[0]["UP"]
        for i in range(len(lp["col_names"])):
            x[lp["col_names"][i]] = [lb[i], ub[i]]
    return x


def make_cf(j, lp):
    A, b, c = parse_matrices(lp)
    n = len(c)
    type = lp["types"][j]

    def cf(x, p):
        s = 0
        for i in range(n):
            s += (
                p[lp["col_names"][i] + "_" + lp["row_names"][j]] * x[lp["col_names"][i]]
            )
        if type == "E" or type == "L":
            return s - p[lp["row_names"][j]]
        elif type == "G":
            return -s + p[lp["row_names"][j]]

    return cf


def parse_equality_constraints(lp):
    eq_con_list = []
    A, b, c = parse_matrices(lp)
    m = len(b)
    for j in range(m):
        if lp["types"][j] == "E":
            cf = make_cf(j, lp)
            eq_con_list += [cf]
    return eq_con_list


def parse_ineq_constraints(lp):
    ineq_con_list = []
    A, b, c = parse_matrices(lp)
    m = len(b)
    for j in range(m):
        if lp["types"][j] == "L" or lp["types"][j] == "G":
            cf = make_cf(j, lp)
            ineq_con_list += [cf]
    return ineq_con_list


def create_lp(path):
    s = time.time()
    lp = parse_lp(path)
    p = parse_parameters(lp, 5)
    x = parse_variables(lp)
    eq_con_list = parse_equality_constraints(lp)
    ineq_con_list = parse_ineq_constraints(lp)

    def obj(x):
        s = 0
        for i in range(len(lp["A"][0, :])):
            s += lp["c"][i] * x[lp["col_names"][i]]
        return s

    ineq_dict = {}
    type = lp["types"]
    j = 0
    for i in range(len(type)):
        if type[i] != "E":
            ineq_dict[j] = i
            j += 1
    eq_dict = {}
    j = 0
    for i in range(len(type)):
        if type[i] == "E":
            eq_dict[j] = i
            j += 1

    cn = lp["col_names"]
    rn = lp["row_names"]
    e = time.time()
    print("Time to create LP:", np.round(e - s, 3), "seconds.")
    return x, p, eq_con_list, ineq_con_list, obj, ineq_dict, eq_dict, cn, rn
