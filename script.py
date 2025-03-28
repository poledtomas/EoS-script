import os
import pandas as pd
import logging
import sys
import warnings
from argparse import ArgumentParser
from logging.handlers import WatchedFileHandler
from pathlib import Path

logger = logging.getLogger("inventer")


def parse_args(argv):
    parser = ArgumentParser(description="Invert EoS data from (mub,T) to (e,nb)")
    parser.add_argument("--logfile", metavar="FILE", help="Log to a specified file")
    parser.add_argument("--loglevel", metavar="LEVEL", default="INFO", help="Log level")

    subparsers = parser.add_subparsers(
        description="valid subcommands", dest="subcommand", required=True
    )
    parser_invert = subparsers.add_parser(
        "invert", help="Invert from (mub,T) to (e,nb)"
    )
    parser_invert.add_argument(
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory",
    )

    parser_invert.add_argument(
        "--input",
        metavar="DIR",
        required=True,
        help="Input directory",
    )
    args = parser.parse_args(argv)
    return args


def configure_logging(filename=None, level=logging.DEBUG):
    if filename:
        handler = WatchedFileHandler(filename, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[handler],
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def open_all_files(directory):
    try:
        enerdens, bardens = None, None
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                logger.info(f"Opening {filename}")
                if "EnerDens_Final_" in filename:
                    enerdens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["mub", "T", "e"]
                    )
                elif "BarDens_Final_" in filename:
                    bardens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["mub", "T", "nb"]
                    )

        if enerdens is None or bardens is None:
            raise ValueError("Missing required data files!")

        return enerdens, bardens
    except Exception as e:
        logger.exception(f"Error: {e}")
        return None, None


def linear_func(e, e2, e3, mu2, mu3):
    result = mu2 + (e - e2) * (mu3 - mu2) / (e3 - e2)

    return result


def inventer(data, value, z):
    logger.info(f"looking for a solution for value = {value}")
    pivot = data.pivot(index="mub", columns="T", values=f"{z}").sort_index()
    T_values = sorted(set(pivot.columns))
    mub_values = sorted(set(pivot.index))

    T_found = []
    mub_found = []

    for T in T_values:
        for mub in range(len(mub_values) - 1):
            mub_low, mub_high = mub_values[mub], mub_values[mub + 1]
            low, high = pivot.loc[mub_values[mub], T], pivot.loc[mub_values[mub + 1], T]

            if low < value and value < high:
                result = linear_func(value, low, high, mub_low, mub_high)
                T_found.append(T)
                mub_found.append(result)

    return T_found, mub_found


def main(argv):
    args = parse_args(argv)
    configure_logging(filename=args.logfile, level=args.loglevel)

    try:
        directory = Path(args.input)
        enerdens, bardens = open_all_files(directory)

        # enerdens["e"] = enerdens["e"] * (enerdens["T"] ** 4)
        # bardens["nb"] = bardens["nb"] * (bardens["T"] ** 3)
        N = 201
        min_e = -0.00999
        max_e = 13.6
        min_nb = -1.369e-14
        max_nb = 0.63

        nb_grid = [min_nb + i * (max_nb - min_nb) / N for i in range(N)]
        e_grid = [min_e + i * (max_e - min_e) / N for i in range(N)]

        for i in e_grid:
            T_e, mub_e = inventer(enerdens, i, "e")
            if not T_e or not mub_e:
                continue
            for j in nb_grid:
                T_nb, mub_nb = inventer(bardens, j, "nb")
                element = set(T_e) & set(T_nb)
                if element:
                    for element in element:
                        index_Te = T_e.index(element)
                        index_Tnb = T_nb.index(element)
                        corresponding_mub_e = mub_e[index_Te]
                        corresponding_mub_nb = mub_nb[index_Tnb]
                        if abs(corresponding_mub_e - corresponding_mub_nb) < 2:
                            print(
                                f"Common element: {element}, mub_e: {corresponding_mub_e}, mub_nb: {corresponding_mub_nb}"
                            )

            # common_elements = set(T) & set(c)
            # for element in common_elements:
            #    index_a = a.index(element)  # Index v poli a
            #    index_c = c.index(element)

            #    corresponding_b = b[index_a]  # Prislusny prvek v poli b
            #    corresponding_d = d[index_c]

            #    if abs(corresponding_b - corresponding_d) < 4:
            #       print(
            #           f"Common element: {element}, b: {corresponding_b}, d: {corresponding_d}"
            #       )

    except Exception:
        logger.exception("Something went wrong.")


if __name__ == "__main__":
    status = main(sys.argv[1:])
    sys.exit(status)
