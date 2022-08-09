# Exercise 7: Function Approximation

## Setup

* Python 3.5+ (for type hints)

Install requirements with the command:
```bash
pip install -r requirements.txt
```

## Complete the code

Fill in the sections marked with `TODO`. The docstrings for each function should help you get started. The starter code is designed to only guide you and you are free to use it as-is or modify it however you like.

### Code structure

- The file `env.py` contains the template code for Four Rooms and you can reuse your implementation from Ex4. If you choose to use the Gym API standard, call `register_env()` once before calling `gym.make()`. 
- There are multiple ways to implement state approximation. An easy way is to use [gym wrappers](https://github.com/openai/gym/blob/master/docs/wrappers.md). These wrappers can modify an environment without redefining or inheriting the environment class. In particular, we will use the [ObservationWrapper](https://github.com/openai/gym/blob/master/gym/core.py#L315-L326) which modifies the observation. See `env.py` for a sample wrapper and how to use it.
- Create your own files for the algorithms and plotting code for each question.
- For Q4, see [http://incompleteideas.net/tiles/tiles3.html] for a Python implementation of tile coding.
