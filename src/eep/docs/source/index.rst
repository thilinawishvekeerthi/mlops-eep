eep documentation
=============================================

Enzyme Efficiency Prediction is an exazyme Package for the prediction of protein properties or attributes (temperature stability, activity efficiency, solubility, etc) using protein sequence information.
It contains multiple sub-packages, each of which is a separate Python package. They are:

- eep: the main package, which contains general functions such as running experiments, building a report, plotting

- eep.models: contains the models used for the prediction of the properties

- eep.generate: contains functions for generating candidate sequences 

- eep.policies: contains the policies used for making sequence suggestions 

- eep.engineer: contains functions for making sequence suggestions 

=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   eep
   eep.models
   eep.policies
   eep.engineer
   eep.generate
   eep.util
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
