# Exercise 5: Temporal-Difference Learning

## Setup

* Python 3.5+ (for type hints)

Install requirements with the command:
```bash
pip install -r requirements.txt
```

## Complete the code

Fill in the sections marked with `TODO`. The docstrings for each function should help you get started. The starter code is designed to only guide you and you are free to use it as-is or modify it however you like.

### Code structure

- Envs: The template code is given in `env.py`. If you choose to use the Gym API standard, call `register_env()` once before calling `gym.make()`. See the docstring in `register_env()` for possible approaches to implementing the windy grid world variants. Some starter code is given for the original WindyGridWorld. You may be able to use some code from the FourRoomsEnv from Ex4.
- Algorithms are given in `algorithms.py`. The algorithms are generally similar in structure. You may be able reuse parts of your code from Ex4.
- Create your own files to plot and run the algorithms for each question.
