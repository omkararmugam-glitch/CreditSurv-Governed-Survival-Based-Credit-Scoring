"""Model registry: every model against the seven approval rules, the background runs
that produce missing evidence, and approval -- the same check as the CLI, made by
the API. The view lives in _registry_view.render so a test can draw it against an
API over its own registry."""

from _registry_view import render

render()
