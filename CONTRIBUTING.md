## Code conventions
This repository uses the following conventions:
- Docstring style: Numpy
    - The concise description of the function/class should come frist
    - In `Parameters` section, the arguments should be explained.
    - In `Returns`, the outputs should be explained.
    - In `Attributes`, the class attributes should be explained.
    - In `Examples`, the example usage should be explained.
- **Annotate types whenever possible** for building the robust and solid code base.
- Code style: PEP8
- Documentation generation: Sphinx
We highly recommend using [Black](https://black.readthedocs.io/en/stable/) formatting and [pylint](https://pypi.org/project/pylint/).

## GitHub Flow
The basic idea is to create feature branches (each associated with a related issue or contribution) for code modifications. Once the code changes are completed, you merge that branch back with the dev branch. The dev branch should always be functional. The dev branch will be merged with main when the codebase is ready for a release. See [this link](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for more details.
Feature branches should be named according to the feature or issue they address, e.g., `feat/123-add-new-feature`, `fix/456-fix-bug-in-module`, etc.

### Commit messages
Commit messages should be clear and concise, following the format:
```
<type>(<scope>): <body>
```
Where:
- `<type>` is the type of change (e.g., feat, fix, docs, style, ref, perf, test)
-- `feat` is for new features
-- `fix` is for bug fixes
-- `docs` is for documentation changes
-- `style` is for code style changes (e.g., formatting, missing semicolons, etc.)
-- `ref` is for code refactoring (e.g., renaming variables, removing unused code, etc.)
-- `perf` is for performance improvements
-- `test` is for adding or updating tests
- `<scope>` is the area of the codebase affected (optional)
- `<body>` is a brief description of the change

### Branching
Always remember to pull so your local repo is up to date:
```
git pull origin main
git status
```
Then create a branch from dev and push:
```
git checkout -b new_feature
git push -u origin new_feature
```

### Adding changes
It is good practice to commit your code changes often.
```
git add changed_file.py
git commit -m "added some new code"
```
To push local commits run
```
git push origin new_feature
```

### Merging
Create a pull request for your feature branch to the dev branch and include a summary of what was done and how it was verified. You can request a review from specific people in the side panel.
