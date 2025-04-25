import numpy as np
from scipy import interpolate

ACCURACY = 1e-6
MAXITER = 100
hbarC = 0.19733
ACCURACY = 1e-6
MAXITER = 100
hbarC = 0.19733

EOS_table_file = "/Users/tomaspolednicek/Desktop/EoSInverter/EOS_filtered.dat"
eos_table = np.loadtxt(EOS_table_file)


hbarC = 0.1973269804
n_T = len(np.unique(eos_table[:, 0]))
n_muB = len(np.unique(eos_table[:, 1]))

T_table = eos_table[:, 0].reshape(n_T, n_muB)
muB_table = eos_table[:, 1].reshape(n_T, n_muB)

factor_4 = 1
factor_3 = 1

ed_table = eos_table[:, 2].reshape(n_T, n_muB) * factor_4  # GeV/fm^3
nB_table = eos_table[:, 3].reshape(n_T, n_muB) * factor_3  # 1/fm^3
P_table = eos_table[:, 4].reshape(n_T, n_muB) * factor_4  # GeV/fm^3
s_table = eos_table[:, 5].reshape(n_T, n_muB) * factor_3  # 1/fm^3

muBarr = np.linspace(np.min(muB_table), np.max(muB_table), n_muB)
Tarr = np.linspace(np.min(T_table), np.max(T_table), n_T)

f_p = interpolate.RectBivariateSpline(Tarr, muBarr, P_table)
f_e = interpolate.RectBivariateSpline(Tarr, muBarr, ed_table)
f_nB = interpolate.RectBivariateSpline(Tarr, muBarr, nB_table)


def binary_search_1d(ed_local, muB_local, Tarr, muBarr):
    iteration = 0
    T_min = Tarr[0]
    T_max = Tarr[-1]
    e_low = f_e(T_min, muB_local)
    e_up = f_e(T_max, muB_local)

    if ed_local < e_low:
        return T_min
    elif ed_local > e_up:
        return T_max
    else:
        T_mid = (T_max + T_min) / 2.0
        e_mid = f_e(T_mid, muB_local)
        abs_err = abs(e_mid - ed_local)
        rel_err = abs_err / abs(e_mid + ed_local + 1e-15)

        while rel_err > ACCURACY and abs_err > ACCURACY * 1e-2 and iteration < MAXITER:
            if ed_local < e_mid:
                T_max = T_mid
            else:
                T_min = T_mid
            T_mid = (T_max + T_min) / 2.0
            e_mid = f_e(T_mid, muB_local)
            abs_err = abs(e_mid - ed_local)
            rel_err = abs_err / abs(e_mid + ed_local + 1e-15)
            iteration += 1
        return T_mid


def binary_search_2d(ed_local, nB_local, Tarr, muBarr):
    iteration = 0

    muB_min = muBarr[0]
    muB_max = muBarr[-1]
    T_min = Tarr[0]
    T_max = Tarr[-1]

    T_max = binary_search_1d(ed_local, muB_min, Tarr, muBarr)
    nB_min = f_nB(T_max, muB_min)
    T_min = binary_search_1d(ed_local, muB_max, Tarr, muBarr)
    nB_max = f_nB(T_min, muB_max)

    if nB_local < nB_min:
        return (T_max, muB_min)
    elif nB_local > nB_max:
        return (T_min, muB_max)
    else:
        muB_mid = (muB_min + muB_max) / 2.0
        T_mid = binary_search_1d(ed_local, muB_mid, Tarr, muBarr)
        nB_mid = f_nB(T_mid, muB_mid)
        abs_err = abs(nB_mid - nB_local)
        rel_err = abs_err / abs(nB_mid + nB_local + 1e-15)

        while rel_err > ACCURACY and abs_err > ACCURACY * 1e-2 and iteration < MAXITER:
            if nB_local < nB_mid:
                muB_max = muB_mid
            else:
                muB_min = muB_mid
            muB_mid = (muB_max + muB_min) / 2.0
            T_mid = binary_search_1d(ed_local, muB_mid, Tarr, muBarr)
            nB_mid = f_nB(T_mid, muB_mid)
            abs_err = abs(nB_mid - nB_local)
            rel_err = abs_err / abs(nB_mid + nB_local)
            iteration += 1
        return T_mid, muB_mid


def invert_EOS_tables(Tarr, muBarr, ed_local, nB_local):
    T_local, muB_local = binary_search_2d(ed_local, nB_local, Tarr, muBarr)
    P_local = f_p(T_local, muB_local)[0]
    return (ed_local, nB_local, P_local, T_local, muB_local)


last_col = -1
Ne = 100
ed_list = np.linspace(1e-2, 2, Ne)
nBmax_list = np.interp(ed_list, ed_table[:, last_col], nB_table[:, last_col])
NnB = 100

output = []
for i, e_i in enumerate(ed_list):
    nB_list = np.linspace(0.0, nBmax_list[i], NnB)
    for j, nB_j in enumerate(nB_list):
        output.append(invert_EOS_tables(Tarr, muBarr, e_i, nB_j))


cleaned_output = []

for row in output:
    e = float(row[0])
    nB = float(row[1])
    P = float(row[2][0])
    T = float(row[3])
    muB = float(row[4])

    cleaned_output.append([e, nB, P, T, muB])

output_array = np.array(cleaned_output)

np.savetxt(
    "id3.dat",
    output_array,
    fmt="%10.6f",
    header="e[GeV/fm^3]  nB[1/fm^3]  P[GeV/fm^3]  T[GeV]  muB[GeV]",
)
