import os
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root


def open_all_files(directory):
    try:
        press, enerdens, bardens = None, None, None
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"Opening {filename}")
                if "Press_Final_PAR_" in filename:
                    press = pd.read_csv(
                        filepath, sep="\t", header=None, names=["T", "mub", "P"]
                    )
                elif "EnerDens_Final_" in filename:
                    enerdens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["T", "mub", "e"]
                    )
                elif "BarDens_Final_" in filename:
                    bardens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["T", "mub", "nb"]
                    )

        if not all([press is not None, enerdens is not None, bardens is not None]):
            raise ValueError("Missing required data files!")

        return press, enerdens, bardens

    except Exception as e:
        print(f"Error: {e}")
        return None


def prepare_interpolators(press, enerdens, bardens):
    """Creates 2D interpolators for e(T, μB), nb(T, μB), and p(T, μB)."""

    e_pivot = enerdens.pivot(index="T", columns="mub", values="e").sort_index()
    nb_pivot = bardens.pivot(index="T", columns="mub", values="nb").sort_index()
    p_pivot = press.pivot(index="T", columns="mub", values="P").sort_index()
    T_vals = e_pivot.index.values
    mub_vals = e_pivot.columns.values

    E_interp = RectBivariateSpline(T_vals, mub_vals, e_pivot.values)
    NB_interp = RectBivariateSpline(T_vals, mub_vals, nb_pivot.values)
    P_interp = RectBivariateSpline(T_vals, mub_vals, p_pivot.values)

    return E_interp, NB_interp, P_interp, T_vals, mub_vals


def inverse_mapping(e_given, nb_given, E_interp, NB_interp, T_vals, mub_vals):
    """Find (T, μB) for given (e, nB) using root-finding."""

    def equations(vars):
        T, mub = vars
        eq1 = E_interp(T, mub)[0, 0] - e_given
        eq2 = NB_interp(T, mub)[0, 0] - nb_given
        return [eq1, eq2]

    T_guess, mub_guess = T_vals[0], mub_vals[0]

    if e_given is not None and nb_given is not None:
        T_guess = T_vals[np.argmin(np.abs(T_vals - e_given))]
        mub_guess = mub_vals[np.argmin(np.abs(mub_vals - nb_given))]

    sol = root(
        equations,
        [T_guess, mub_guess],
        method="lm",
        options={"xtol": 1e-3, "maxiter": 1000},
    )

    if sol.success:
        return sol.x
    else:
        print("Root finding failed. Solver did not converge.")
        print(f"Solver message: {sol.message}")
        print(f"Last attempt: T = {sol.x[0]}, μB = {sol.x[1]}")
        return None


def pressure_from_enb(
    e_given, nb_given, E_interp, NB_interp, P_interp, T_vals, mub_vals
):
    """Compute p(e, nB) using interpolation."""
    result = inverse_mapping(e_given, nb_given, E_interp, NB_interp, T_vals, mub_vals)
    if result is None:
        print("Failed to find T and μB. Returning NaN for pressure.")
        return np.nan
    T, mub = result
    return P_interp(T, mub)[0, 0]


def main():
    directory_path = "/Users/tomaspolednicek/Desktop/EoS-script/data/"
    press, enerdens, bardens = open_all_files(directory_path)

    E_interp, NB_interp, P_interp, T_vals, mub_vals = prepare_interpolators(
        press, enerdens, bardens
    )

    e_input = []  # List to store example energy densities
    nb_input = []  # List to store baryon densities

    e_min = enerdens["e"].min()
    e_max = enerdens["e"].max()
    nb_min = bardens["nb"].min()
    nb_max = bardens["nb"].max()
    Bins = 400

    de = (e_max - e_min) / Bins
    dnb = (nb_max - nb_min) / Bins
    for i in range(Bins):
        e_input.append(e_min + i * de)
        nb_input.append(nb_min + i * dnb)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "EoS.dat")
    print(output_filename)
    with open(output_filename, "a") as fout:
        for e in e_input:
            for nb in nb_input:
                result = inverse_mapping(e, nb, E_interp, NB_interp, T_vals, mub_vals)

                if result is not None:
                    T_result, mub_result = result
                    p_result = pressure_from_enb(
                        e, nb, E_interp, NB_interp, P_interp, T_vals, mub_vals
                    )

                    fout.write(
                        f"{e:.6f}\t{nb:.6f}\t{T_result:.6f}\t{mub_result:.6f}\t{p_result:.6f}\n"
                    )

                    # print(f"T = {T_result:.2f} MeV, μB = {mub_result:.2f} MeV, p = {p_result:.2f} MeV/fm^3")
                else:
                    print(f"Failed to calculate values for e = {e:.2f}, nB = {nb:.2f}.")

    fout.close()


if __name__ == "__main__":
    main()
