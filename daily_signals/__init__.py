"""Daily crypto trading signals product built alongside TensorTrade.

This package is intentionally dependency-light: the daily pipeline
(fetch -> strategy voices -> ensemble -> paper portfolio -> reports -> email)
requires only pandas, numpy, requests, jinja2, pyyaml and matplotlib.

Heavy dependencies (ray/tensorflow/torch/tensortrade) are only imported
lazily inside :mod:`daily_signals.rl` when the optional RL voice is enabled.
Nothing at this top level may import them.
"""

__version__ = "0.1.0"
