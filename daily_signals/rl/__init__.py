"""Optional RL voice for the daily signals ensemble.

Nothing in this package is imported by the light pipeline unless
``rl.enabled: true`` — and even then failures degrade to an "unavailable"
voice rather than breaking the daily run. Requires `pip install -e ".[rl]"`
(ray[rllib] + torch) plus the tensortrade package itself.
"""
