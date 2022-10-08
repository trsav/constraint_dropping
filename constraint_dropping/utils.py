import numpy as np
import os
from pysmps import smps_loader as smps
import logging
from cmath import inf
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

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
            if A[j, i] != 0:
                s += (
                    p[lp["col_names"][i] + "_" + lp["row_names"][j]]
                    * x[lp["col_names"][i]]
                )
        if b[j] != 0:
            b_val = p[lp["row_names"][j]]
        else:
            b_val = 0

        if type == "E" or type == "L":
            return s - b_val
        elif type == "G":
            return -s + b_val

    return [cf, type]


def parse_constraints(lp):
    eq_con_list = []
    ineq_con_list = []
    A, b, c = parse_matrices(lp)
    for j in range(len(b)):
        if max(A[j, :]) == 0 and min(A[j, :]) == 0:
            pass
        cf, type = make_cf(j, lp)
        if type == "E":
            eq_con_list += [cf]
        else:
            ineq_con_list += [cf]

    return ineq_con_list, eq_con_list, A, b, c


def create_lp(path):
    lp = parse_lp(path)
    p = parse_parameters(lp, 1)
    x = parse_variables(lp)
    ineq_con_list, eq_con_list, A, b, c = parse_constraints(lp)

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
    return (
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
    )


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
        y[i] = (x[i] - min_x) / (max_x - min_x)
    return y


def plot_result():
    k_list = [
        "Robust Constraint Violations (%)",
        "Maximum Constraint Violation",
        "Nominal Time Factor",
        "Upper Level Problem Size",
    ]
    x = "Iteration"
    fig, axs = plt.subplots(1, len(k_list), figsize=(12, 3))
    fig.set_constrained_layout(True)
    for filename in os.listdir("outputs"):
        if filename[-3:] != "csv":
            continue
        if filename.split("_")[-1].split(".")[0] == "standard":

            col = "k"
        else:
            col = "r"

        filename = "outputs/" + filename
        df = pd.read_csv(filename)
        total_cons = (
            df["Upper Level Problem Size"][0] - df["Robust Constraint Violations"][0]
        )

        if len(df[x]) <= 1:
            continue
        for i in range(len(k_list)):
            axs[i].grid(True, alpha=0.3)
            k = k_list[i]
            if k != "Time Factor":
                xp = df[x][:-1]
                kp = df[k][:-1]
            else:
                xp = df[x]
                kp = df[k]

            if k == "Upper Level Problem Size":
                kp = ((kp - total_cons) / total_cons) + 1
                axs[i].set_ylabel("Normalized Upper Level Problem Size")
                # kp = normalize_vector(kp)
                # axs[i].set_ylabel('Normalized '+k)
            else:
                axs[i].set_ylabel(k)
            axs[i].plot(xp, kp, c=col, lw=0.75)
            # axs[i].set_xlim(1,11)
            axs[i].set_xlabel(x)
            # axs[i].set_xticks(x_ticks,x_ticks)
            if i == 1 or i == 0:
                xp = xp.values
                kp = kp.values
                # axs[i].plot([xp[-1],xp[-1]],[kp[-1],0],c=col,alpha=0.1)
                axs[i].set_yscale("log")
    plt.savefig("outputs/all_outputs.png", dpi=600)

    return


plot_result()
