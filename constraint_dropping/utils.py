import numpy as np
from pysmps import smps_loader as smps
import logging
from cmath import inf
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt

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
    print("Reading MPS from", path)
    s = time.time()
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
    print("Successfully read MPS in ", np.round(time.time() - s, 3), " seconds")
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

    print("Parsing parameters")
    for i in tqdm(range(n)):
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

    return [cf, type]


def parse_constraints(lp):
    eq_con_list = []
    ineq_con_list = []
    A, b, c = parse_matrices(lp)
    for j in range(len(b)):
        cf, type = make_cf(j, lp)
        if type == "E":
            eq_con_list += [cf]
        else:
            ineq_con_list += [cf]

    return ineq_con_list, eq_con_list


def create_lp(path):
    lp = parse_lp(path)
    p = parse_parameters(lp, 1)
    x = parse_variables(lp)
    ineq_con_list, eq_con_list = parse_constraints(lp)

    def obj(x):
        s = 0
        for i in range(len(lp["A"][0, :])):
            s += lp["c"][i] * x[lp["col_names"][i]]
        return s

    ineq_dict = {}
    eq_dict = {}
    type = lp["types"]
    jin = 0
    jeq = 0
    for i in range(len(type)):
        if type[i] != "E":
            ineq_dict[jin] = i
            jin += 1
        else:
            eq_dict[jeq] = i
            jeq += 1

    cn = lp["col_names"]
    rn = lp["row_names"]
    return x, p, eq_con_list, ineq_con_list, obj, ineq_dict, eq_dict, cn, rn


def names_to_list():
    names = open("lp_files/lp_names", "rb").readlines()
    for i in range(len(names)):
        names[i] = str(names[i])
        names[i] = names[i].split("""b'""")[-1]
        names[i] = names[i].split(" ")[0]
        names[i] = names[i].split("""\\""")[0]
    return names


def normalize_vector(x):
    y = np.zeros_like(x)
    min_x = min(x)
    max_x = max(x)
    for i in range(len(y)):
        y[i] = 100 * (x[i] - min_x) / (max_x - min_x)
    return y


def plot_result(mps_list):
    k_list = [
        "Robust Constraint Violations",
        "Time (s)",
        "Maximum Constraint Violation",
    ]
    x = "Iteration"
    fig, axs = plt.subplots(2, len(k_list), figsize=(12, 6))
    fig.set_constrained_layout(True)
    if type(mps_list) == str:
        mps_list = [mps_list]
    for mps in mps_list:
        df = pd.read_csv("outputs/" + mps + ".csv")
        for i in range(len(k_list)):
            k = k_list[i]
            axs[0, i].plot(df[x], df[k], c="k")

            k_norm = normalize_vector(df[k])
            axs[1, i].plot(df[x], k_norm, c="k")

            axs[0, i].set_ylabel(k)
            axs[1, i].set_ylabel("Normalised " + str(k))
            for j in range(2):
                axs[j, i].set_xlabel(x)
                axs[j, i].set_xticks(df[x], df[x])

    plt.savefig("outputs/all_outputs.png", dpi=600)

    return


plot_result(["agg3", "afiro", "agg3", "beaconfd", "25fv47", "adlittle"])
