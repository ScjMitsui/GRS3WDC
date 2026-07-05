"""GRS3WDC: Granular Rough Sets Three-Way Decision Classifier.

Reference implementation for "Three-way decision with granular rough sets"
(Luo, Hu, Shi, Yao; Applied Soft Computing 180, 2025).
"""

from grs3wdc.evaluation import EvaluationResult, SummaryStats, repeated_holdout

__all__ = ["EvaluationResult", "SummaryStats", "repeated_holdout"]
