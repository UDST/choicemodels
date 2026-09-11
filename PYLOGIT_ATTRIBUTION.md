# PyLogit attribution

ChoiceModels' flexible multinomial-logit implementation retains the model-
specification conventions of PyLogit. The design-matrix construction in
`choicemodels/pylogit_compat.py` is adapted from PyLogit 1.0.1.

PyLogit was created by Timothy A. Brathwaite. Its original contributor record
is preserved in `PYLOGIT_CONTRIBUTORS.txt`. The incorporated work was taken from revision
`cffc9c523b5368966ef2481c7dc30f0a5d296de8` of
https://github.com/timothyb0912/pylogit. It remains subject to the BSD 3-Clause
license reproduced in `licenses/PYLOGIT_LICENSE.txt`.

ChoiceModels contains only the multinomial-logit functionality it uses. It
does not incorporate PyLogit's nested, mixed, asymmetric, bootstrap, or other
model families. Changes made in ChoiceModels are recorded in its changelog and
Git history so that relevant improvements can be evaluated for PyLogit.
