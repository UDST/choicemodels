Thanks for using ChoiceModels! 

This is an open source project that's part of the Urban Data Science Toolkit. Development and maintenance is a collaboration between UrbanSim Inc and other contributors. 

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

- Add [tests](https://github.com/UDST/choicemodels/tree/main/tests) if possible!

- Open a pull request to the `UDST/choicemodels` main branch, including a writeup of your changes -- take a look at some of the closed PR's for examples. (Before v0.3, development happened on a `dev` branch; as of the v0.3 release, `main` is the integration branch and `dev` is retired.)

- Automated checks run on every pull request: the test suite against the oldest and newest supported Python and dependency versions, code quality checks, a package build, and a documentation build

- Current maintainers will review the code, suggest changes, and hopefully merge it!


## Updating the version number:

- Each pull request that changes substantive code should increment the development version number, e.g. from `0.2.dev7` to `0.2.dev8`, so that users know exactly which version they're running

- It works best to do this just before merging (in case other PR's are merged first, and so you know the release date for the changelog and documentation)

- There are two places where the version number needs to be changed (`pyproject.toml` reads it from the package): 
  - `choicemodels/__init__.py`
  - `docs/source/index.rst`

- Please also add a section to `CHANGELOG.md` describing the changes!


## Updating the documentation: 

- See instructions in `docs/README.md`


## Preparing a release:

- Make a new branch for release prep

- Update the version number and `CHANGELOG.md`
  - `choicemodels/__init__.py`
  - `docs/source/index.rst`

- Make sure all the tests are passing, and check if updates are needed to `README.md` or to the documentation

- Open a pull request to the main branch, and merge it once the checks pass

- Publish the release on GitHub: create a tag on the merge commit named with a `v` prefix (e.g. `v0.3` for version `0.3`), and use the changelog text as the release notes. Publishing the release starts the `Publish` workflow described below

- For anything more than a trivial release, do a dry run first with a release candidate: set the version to e.g. `0.3rc1`, tag it `v0.3rc1`, and mark the GitHub release as a pre-release. Pip ignores pre-releases unless asked for them (`pip install --pre choicemodels==0.3rc1`), and the Conda Forge bots ignore them too, so this is a safe way to test the whole process. There's no need to delete the release candidate from PyPI afterward. Then repeat with the final version number

- After the release, rebuild and publish the documentation (see `docs/README.md`)


## Distributing a release on PyPI (for pip installation):

- Publishing is automated by the `Publish` GitHub Actions workflow (`.github/workflows/publish.yml`), which runs when a release is published on GitHub. It builds the source distribution and wheel, checks them with `twine check --strict`, installs the wheel and runs the test suite against it, confirms that the package version matches the release tag, and then uploads the files to PyPI using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no PyPI credentials are stored on GitHub

- The upload step runs in the repository's `pypi` deployment environment, which requires approval from a maintainer: once the build job succeeds, the workflow pauses until a reviewer approves the deployment from the workflow run page. Merging to `main` never publishes anything

- Check https://pypi.org/project/choicemodels/ for the new version, and try `pip install choicemodels` in a fresh environment

- One-time setup, in case it needs to be repeated: a PyPI owner of the project registers the trusted publisher at https://pypi.org/manage/project/choicemodels/settings/publishing/ with owner `UDST`, repository `choicemodels`, workflow `publish.yml`, and environment `pypi`; and a repository admin creates the `pypi` environment at https://github.com/UDST/choicemodels/settings/environments with required reviewers

- Manual fallback, if the workflow can't be used: register an account at https://pypi.org with two-factor authentication enabled, ask one of the current maintainers to add you to the project, and create an API token scoped to the ChoiceModels project. Then `pip install build twine`, delete any old files in `dist/`, and run `python -m build`, `twine check --strict dist/*`, and `twine upload dist/*`, entering `__token__` as the username and the token as the password


## Distributing a release on Conda Forge (for conda installation):

- The [conda-forge/choicemodels-feedstock](https://github.com/conda-forge/choicemodels-feedstock) repository controls the Conda Forge release, including which GitHub users have maintainer status for the feedstock

- Conda Forge bots usually detect new releases on PyPI within a few hours and open a pull request to update the feedstock, which a current feedstock maintainer needs to review and merge

- Before merging, check that the run requirements and the Python version floor in `recipe/meta.yaml` still match `pyproject.toml`; the bot only updates the version and hash. Additional changes can be pushed to the bot's branch, for example to update the requirements or the list of maintainers

- You can also fork the feedstock and open a pull request manually, updating the version number and pasting the new hash of the `.tar.gz` file uploaded to PyPI (available on the pypi.org project page). It seems like this must be done from a personal account (not a group account like UDST) so that the bots can be granted permission for automated cleanup

- Check https://anaconda.org/conda-forge/choicemodels for the new version (may take a few minutes for it to appear)
