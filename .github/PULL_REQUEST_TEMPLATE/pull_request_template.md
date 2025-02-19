## Summary

Describe your changes here specifying if the change is a bug fix, enhancement, new feature, etc. Mention relevent issues or discussion posts as necessary.

## Test Plan

Describe how you tested and verified your changes here (changes captured in existing tests, built and ran new tests, etc.).

## Before Submitting
- [ ] Check for bugs/errors
    - [ ] Run example notebooks
        - `examples/`
        - Ensure all notebooks run successfully.
    - [ ] Write or update unit tests in `tests/`
    - [ ] Run Pytest: `pytest`
        - Ensure all tests pass successfully
- [ ] Run Pylint: `pylint --rcfile=.pylintrc torchsig`
    - [ ] Score > 9/10
    - [ ] Code conforms with [PEP 8 Python Style Guide](https://peps.python.org/pep-0008/)
    - [ ] [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) used to document code
        - This ensures our documentation stays up to date
- [ ] Added contributers to PR