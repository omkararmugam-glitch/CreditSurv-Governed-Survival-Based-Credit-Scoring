"""Model registry: every model against the seven approval rules, the background runs
that produce missing evidence, and approval -- the same check as the CLI.

The view lives in _registry_view.render so a test can draw it against its own
registry without touching config/models.yaml.
"""

from _common import CONFIG_PATH
from _registry_view import render
from creditsurv.config import load_config

render(load_config(CONFIG_PATH))
