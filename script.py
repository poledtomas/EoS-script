import os
import pandas as pd
import logging
import sys
import warnings
import multiprocessing
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
        "--output", metavar="DIR", required=True, help="Output directory"
    )
    parser_invert.add_argument(
        "--input", metavar="DIR", required=True, help="Input directory"
    )

    return parser.parse_args(argv)


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
    return mu2 + (e - e2) * (mu3 - mu2) / (e3 - e2)


def inventer(data, value, z):
    logger.info(f"Looking for a solution for value = {value}")
    pivot = data.pivot(index="mub", columns="T", values=f"{z}").sort_index()
    T_values = sorted(set(pivot.columns))
    mub_values = sorted(set(pivot.index))

    T_found = []
    mub_found = []

    for T in T_values:
        for mub in range(len(mub_values) - 1):
            mub_low, mub_high = mub_values[mub], mub_values[mub + 1]
            low, high = pivot.loc[mub_values[mub], T], pivot.loc[mub_values[mub + 1], T]

            if low < value < high:
                result = linear_func(value, low, high, mub_low, mub_high)
                T_found.append(T)
                mub_found.append(result)

    return T_found, mub_found


def process_e_value(args):
    """Function to process a single value of e_grid (parallelized)."""
    i, enerdens, bardens, nb_grid = args  # Unpack arguments
    results = []
    T_e, mub_e = inventer(enerdens, i, "e")
    if not T_e or not mub_e:
        return results  # Skip if no valid results

    for j in nb_grid:
        T_nb, mub_nb = inventer(bardens, j, "nb")
        common_elements = set(T_e) & set(T_nb)

        if common_elements:
            for element in common_elements:
                index_Te = T_e.index(element)
                index_Tnb = T_nb.index(element)
                corresponding_mub_e = mub_e[index_Te]
                corresponding_mub_nb = mub_nb[index_Tnb]

                if abs(corresponding_mub_e - corresponding_mub_nb) < 2:
                    results.append(
                        f"e: {i}, nb: {j}, Te: {element}, Tnb: {element}, mub_e: {corresponding_mub_e}, mub_nb: {corresponding_mub_nb}"
                    )
    return results


def main(argv):
    args = parse_args(argv)
    configure_logging(filename=args.logfile, level=args.loglevel)

    try:
        directory = Path(args.input)
        enerdens, bardens = open_all_files(directory)

        N = 100
        min_e, max_e = -0.00999, 13.6
        min_nb, max_nb = -1.369e-14, 0.63

        nb_grid = [min_nb + i * (max_nb - min_nb) / N for i in range(N)]
        e_grid = [min_e + i * (max_e - min_e) / N for i in range(N)]

        # Prepare argument tuples for multiprocessing
        task_args = [(i, enerdens, bardens, nb_grid) for i in e_grid]

        # Use multiprocessing to process e_grid values in parallel
        num_workers = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(process_e_value, task_args)

        # Flatten and print results safely
        for res in results:
            for line in res:
                logger.info(line)

    except Exception:
        logger.exception("Something went wrong.")


if __name__ == "__main__":
    status = main(sys.argv[1:])
    sys.exit(status)
