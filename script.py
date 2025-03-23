import os
import pandas as pd
from scipy.interpolate import RectBivariateSpline


def open_all_files(directory):
    try:
        enerdens, bardens = None, None
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"Opening {filename}")
                if "EnerDens_Final_" in filename:
                    enerdens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["T", "mub", "e"]
                    )
                elif "BarDens_Final_" in filename:
                    bardens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["T", "mub", "nb"]
                    )

        if not all([enerdens is not None, bardens is not None]):
            raise ValueError("Missing required data files!")

        return enerdens, bardens
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def prepare_interpolators(enerdens, bardens):
    e_pivot = enerdens.pivot(index="T", columns="mub", values="e").sort_index()
    nb_pivot = bardens.pivot(index="T", columns="mub", values="nb").sort_index()

    T = sorted(set(e_pivot.index) & set(nb_pivot.index))
    mub = sorted(set(e_pivot.columns) & set(nb_pivot.columns))
    e_pivot = e_pivot.loc[T, mub]

    nb_pivot = nb_pivot.loc[T, mub]

    E_interp = RectBivariateSpline(T, mub, e_pivot.values)
    NB_interp = RectBivariateSpline(T, mub, nb_pivot.values)

    return E_interp, NB_interp, T, mub


def find_eos(E_interp, NB_interp, T_values, mub_values, e_value, nb_value):
    print("Hledáme řešení pro e =", e_value, ", nb =", nb_value)

    for mub in mub_values:
        T_found = None
        T_nb_found = None

        for T in range(len(T_values) - 1):
            T_low, T_high = T_values[T], T_values[T + 1]
            e_low, e_high = E_interp(T_low, mub)[0][0], E_interp(T_high, mub)[0][0]
            if e_low <= e_value <= e_high:
                T_min, T_max = T_low, T_high
                while abs(T_max - T_min) > 1e-6:
                    T_mid = (T_min + T_max) / 2
                    e_mid = E_interp(T_mid, mub)[0][0]
                    if (e_mid - e_value) * (e_low - e_value) < 0:
                        T_max = T_mid
                    else:
                        T_min = T_mid
                T_found = (T_min + T_max) / 2
                break

        for T in range(len(T_values) - 1):
            T_low, T_high = T_values[T], T_values[T + 1]

            nb_low, nb_high = NB_interp(T_low, mub)[0][0], NB_interp(T_high, mub)[0][0]
            if nb_low <= nb_value <= nb_high:
                T_min, T_max = T_low, T_high
                while abs(T_max - T_min) > 1e-6:
                    T_mid = (T_min + T_max) / 2
                    nb_mid = NB_interp(T_mid, mub)[0][0]
                    if (nb_mid - nb_value) * (nb_low - nb_value) < 0:
                        T_max = T_mid
                    else:
                        T_min = T_mid

                T_nb_found = (T_min + T_max) / 2
                break

        if T_found is not None and T_nb_found is not None:
            print(f"mub = {mub}, T_found = {T_found}, T_nb_found = {T_nb_found}")
            if abs(T_nb_found - T_found) < 2.0:
                print("Řešení nalezeno:", T_found, mub)
                return T_found, mub

    print("Žádné řešení nebylo nalezeno.")
    return None


def main():
    directory = "/Users/tomaspolednicek/Desktop/EoS-script/data"
    enerdens, bardens = open_all_files(directory)
    E_interp, NB_interp, T, mub = prepare_interpolators(enerdens, bardens)
    min = 0
    max = 13
    N = 400
    dp = (max - min) / N
    e_grid = []
    nb_grid = []

    for i in range(N):
        e_grid.append(min + (i + 0.5) * dp)
        nb_grid.append(min + (i + 0.5) * dp)

    if enerdens is not None and bardens is not None:
        # e_value = 0.1748480499691949
        # nb_value = 0.0000000000000700
        e_value = 11.68
        nb_value = 0.39

        find_eos(E_interp, NB_interp, T, mub, e_value, nb_value)

        # result = find_eos(E_interp, NB_interp, T, mub, e_value, nb_value)

        # if result[0] is not None:
        #    print(f"Matching values found: T = {result[0]}, mub = {result[1]}")
        # else:
        #    print("No matching values found.")


if __name__ == "__main__":
    main()
