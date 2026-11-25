# Recorded DaVinci Resolve behaviour

Each `*.json` here is a probe report from a real Resolve: in the Glitcher panel, click
**Check this Resolve**, then copy `resolve_probe.json` from the Glitcher Resolve folder to
`tests/resolve_contract/<resolve version>-<os>.json`.

`test_resolve_plugin.py::test_fake_matches_recorded_resolve` runs the same probe against the
fake Resolve the plugin is tested with, and fails if the fake answers any question differently
from the recorded Resolve: how sequences import, how clips are placed and trimmed, which inputs
Fusion's nodes have, how renders are named. The fake is only allowed to be as right as the
recordings say.
