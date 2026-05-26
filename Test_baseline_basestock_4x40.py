#!/usr/bin/env python
"""
Baseline Evaluation Script: (s,S) Heuristic Policy -- 4 DC x 40 Retailer Scale
==============================================================================

Thin wrapper around `Test_baseline_basestock.py` that points the environment
to the 4x40 topology config. All evaluation logic, metrics, plots, CSV/JSON
output formats, and default episode/episode-length settings are inherited
unchanged from the reference script.

Usage (all arguments optional):
    python Test_baseline_basestock_4x40.py
    python Test_baseline_basestock_4x40.py --num_episodes 5 --episode_length 365
    python Test_baseline_basestock_4x40.py \\
        --s_dc 100 --S_dc 170 --s_retailer 3 --S_retailer 10 \\
        --experiment_name "basestock_4x40"
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from Test_baseline_basestock_2x15 import BaseStockEvaluator, parse_args


CONFIG_PATH_DEFAULT = 'configs/multi_dc_4x40_config.yaml'


class BaseStockEvaluator4x40(BaseStockEvaluator):
    """4x40 topology-specific evaluator. Only overrides the output CSV name."""
    RESULTS_CSV_NAME = 'results_ss_heuristic_4x40.csv'


def main():
    args = parse_args()

    if args.config_path == 'configs/multi_dc_config.yaml':
        args.config_path = CONFIG_PATH_DEFAULT

    print('\n=== (s,S) Heuristic Configuration (4x40) ===')
    print(f'  Config file              : {args.config_path}')
    print(f'  DC       (s, S) per SKU  : ({args.s_dc}, {args.S_dc})')
    print(f'  Retailer (s, S) per SKU  : ({args.s_retailer}, {args.S_retailer})')
    print('============================================\n')

    evaluator = BaseStockEvaluator4x40(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
