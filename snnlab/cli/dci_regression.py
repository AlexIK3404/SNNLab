from __future__ import annotations

import argparse

from snnlab.regression import run_dci_notebook_regression


def build_parser() -> argparse.ArgumentParser:
    """
    Builds CLI arguments for the DCI notebook regression check.

    Создаёт CLI-аргументы для regression-проверки DCI относительно блокнота.
    """
    parser = argparse.ArgumentParser(
        description="Compare Stage-1 DCI with the frozen clean-notebook reference"
    )
    parser.add_argument("--locale", choices=("en", "ru"), default="en")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_dci_notebook_regression()

    if args.locale == "ru":
        print("=" * 72)
        print("DCI REGRESSION: ФРЕЙМВОРК VS CLEAN-БЛОКНОТ")
        print("=" * 72)
        for check in report.checks:
            status = "OK" if check.passed else "FAIL"
            print(f"[{status:4}] {check.name:<28} {check.details}")
        print("-" * 72)
        print("ИТОГ:", "ПРОЙДЕНО" if report.passed else "ЕСТЬ РАСХОЖДЕНИЯ")
    else:
        print("=" * 72)
        print("DCI REGRESSION: FRAMEWORK VS CLEAN NOTEBOOK")
        print("=" * 72)
        for check in report.checks:
            status = "OK" if check.passed else "FAIL"
            print(f"[{status:4}] {check.name:<28} {check.details}")
        print("-" * 72)
        print("RESULT:", "PASSED" if report.passed else "MISMATCHES DETECTED")

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
