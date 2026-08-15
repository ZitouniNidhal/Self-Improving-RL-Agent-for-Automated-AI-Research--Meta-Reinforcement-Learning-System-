# Contributing

Contributions are welcome! This project is intentionally kept small and
dependency-light (NumPy only for the core library) so it stays easy to
read end-to-end -- please keep that spirit in mind for PRs.

## Getting set up

```bash
git clone https://github.com/ZitouniNidhal/Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-.git
cd Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Running tests

```bash
cd tests
python -m pytest -q
```

## Style

* Keep new core-library code dependency-free (NumPy only) unless there's
  a strong reason otherwise -- discuss in an issue first.
* Prefer small, well-commented functions over clever one-liners; this is
  meant to be a teaching-quality reference implementation.
* Add or update tests for any behavioural change.

## Ideas for contributions

See the Roadmap section in `README.md` for larger feature ideas (PPO
inner loop, real hyperparameter-search backends, richer task
distributions, visualization dashboard, etc.). Small fixes, docs
improvements, and additional tests are always appreciated too.

## Reporting issues

Please include: Python version, OS, the command you ran, and the full
traceback / unexpected output.
