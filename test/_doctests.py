import doctest

verbose = True
RUN_ALL_TESTS = False
# ========== examples ==========

# testing a file
# doctest.testfile("../src/eep/models/gpytorch_models.py", verbose=True)

# testing a module
# import eep.models.gpytorch_models as gpytorch_models
# doctest.testmod(gpytorch_models, verbose=True)

# ========== doc tests ================
if RUN_ALL_TESTS:
    # ========== gpytorch module ==========
    import eep.models.gpytorch_models as gpytorch_models

    doctest.testmod(gpytorch_models, verbose=verbose)

    # ========== pretrained module ==========
    import eep.models.pretrained as pretrained

    doctest.testmod(pretrained, verbose=verbose)

    # ========== pytorch_model module ==========
    import eep.models.pytorch_models as pytorch_models

    doctest.testmod(pytorch_models, verbose=verbose)

    # ========== utils module ==========
    import eep.models.utils as utils

    doctest.testmod(utils, verbose=verbose)

    # ========== kernel_regression_pytorch module ==========
    import eep.models.kernel_regression_pytorch as kernel_regression_pytorch

    doctest.testmod(kernel_regression_pytorch, verbose=verbose)


# ========== generate package ===========
if True:
    # ========== ancestor module ==========
    from eep.generate import ancestor

    doctest.testmod(ancestor, verbose=verbose)
