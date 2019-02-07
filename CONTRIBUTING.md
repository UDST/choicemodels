Thanks for using ChoiceModels! 

This is an open source project that's part of the Urban Data Science Toolkit. Development and maintenance is a collaboration between UrbanSim Inc and U.C. Berkeley's Urban Analytics Lab. 

You can contact Sam Maurer, the lead developer, at `maurer@urbansim.com`.


## If you have a problem:

- Take a look at the [open issues](https://github.com/UDST/choicemodels/issues) and [closed issues](https://github.com/UDST/choicemodels/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a related discussion

- Open a new issue describing the problem -- if possible, include any error messages, the operating system and version of python you're using, and versions of any libraries that may be relevant


## Feature proposals:

- Take a look at the [open issues](https://github.com/UDST/choicemodels/issues) and [closed issues](https://github.com/UDST/choicemodels/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a related discussion

- Post your proposal as a new issue, so we can discuss it (some proposals may not be a good fit for the project)


## Contributing code:

- Create a new branch of `UDST/choicemodels`, or fork the repository to your own account

- Make your changes, following the existing styles for code and inline documentation

- Add [tests](https://github.com/UDST/choicemodels/tree/master/tests) if possible!

- Open a pull request to the `UDST/choicemodels` master branch, including a writeup of your changes -- take a look at some of the closed PR's for examples

- Current maintainers will review the code, suggest changes, and hopefully merge it!


## Updating the version number:

- Each pull request that changes substantive code should increment the development version number, e.g. from `0.2.dev7` to `0.2.dev8`, so that users know exactly which version they're running

- It works best to do this just before merging (in case other PR's are merged first, and so you know the release date for the changelog and documentation)

- There are three places where the version number needs to be changed: 
  - `setup.py`
  - `choicemodels/__init__.py`
  - `docs/source/index.rst`

- Please also add a section to `CHANGELOG.md` describing the changes!


## Updating the documentation: 

- See instructions in `docs/README.md`


## Preparing a production release:

- Make a new branch for release prep

- Update the version number and `CHANGELOG.md`

- Make sure all the tests are passing, and check if updates are needed to `README.md` or to the documentation

- Open a pull request to the master branch and merge it

- Tag the release on Github


## Distributing a release on PyPI (for pip installation):

- Register an account at https://pypi.org, ask one of the current maintainers to add you to the project, and `pip install twine`

- Run `python setup.py sdist bdist_wheel --universal`

- This should create a `dist` directory containing two package files -- delete any old ones before the next step

- Run `twine upload dist/*` -- this will prompt you for your pypi.org credentials

- Check https://pypi.org/project/choicemodels/ for the new version


## Distributing a release on Conda Forge (for conda installation):

- Make a fork of the [conda-forge/choicemodels-feedstock](https://github.com/conda-forge/choicemodels-feedstock) repository -- there may already be a fork in udst

- Edit `recipe/meta.yaml`: 
  - update the version number
  - paste a new hash matching the tar.gz file that was uploaded to pypi (it's available on the pypi.org project page)

- Check that the run requirements still match `requirements.txt`

- Open a pull request to the `conda-forge/choicemodels-feedstock` master branch

- Automated tests will run, and after they pass one of the current project maintainers will be able to merge the PR -- you can add your Github user name to the maintainers list in `meta.yaml` for the next update

- Check https://anaconda.org/conda-forge/choicemodels for the new version (may take a few minutes for it to appear)
