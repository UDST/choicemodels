Thanks for using ChoiceModels! This is an open source project that's part of the Urban Data Science Toolkit. Development and maintenance is a collaboration between UrbanSim Inc and U.C. Berkeley's Urban Analytics Lab. You can contact Sam Maurer, the lead developer, at `maurer@urbansim.com`.

### If you encounter an error or find a bug:

- Take a look at the [open issues](https://github.com/UDST/choicemodels/issues) and [closed issues](https://github.com/UDST/choicemodels/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a discussion of the problem

- Open a new issue describing the problem: circumstances, error messages, operating system you are using, version of python, and version of any libraries that may be relevant

### If you have a feature proposal:

- Take a look at the [open issues](https://github.com/UDST/choicemodels/issues) and [closed issues](https://github.com/UDST/choicemodels/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a discussion of the topic

- Post your proposal as a new issue, so we can discuss it (some proposals may not be a good fit for the project)

### Adding a feature or fixing a bug:

- Create a new branch of UDST/choicemodels, or fork the repository to your own account

- Make your changes, adhering to the existing styles for coding, commenting, and especially the documentation strings at the beginning of functions

- Add [tests](https://github.com/UDST/choicemodels/tree/master/tests) if possible

- When you're ready to begin code review, open a pull request to the UDST/choicemodels master branch

- The pull request writeup should be clear and thorough, to facilitate code review, documentation, and release notes (see [example here](https://github.com/UDST/choicemodels/pull/43)). First, briefly summarize the changes, referencing any associated issue threads. Then describe the changes in more detail: implementation, usage, performance, and anything else that's relevant

- Make note in the pull request writeup of any API changes (class/method/function names, parameters, and behavior), particularly changes that could affect users' existing code

- Each substantial pull request should increment the development version number, e.g. from 0.2.dev7 to 0.2.dev8

- If incrementing the version number: (1) update `setup.py`, (2) update `choicemodels/__init__.py`, (3) add a section to `CHANGELOG.md`, and (4) add the version number to the beginning of the pull request name

### Preparing a production release:

- Create a branch for release prep

- Make sure all the tests are passing

- Update the version number (e.g. from 0.2.dev8 to 0.2) in `setup.py` and `choicemodels/__init__.py`

- Update `CHANGELOG.md`, collapsing development release sections into a single, reorganized list

- Check if updates are needed to `README.md` and to the documentation source files

- Rebuild the documentation webpages (DETAILS TK)

- Open a pull request to the master branch

- Merge the pull request

- Tag the release on Github

- Update the Python Package Index (DETAILS TK)

- Update the UDST Conda channel (DETAILS TK)
