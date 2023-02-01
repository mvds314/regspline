from setuptools import setup

setup(
    name="regspline",
    version="23.1",
    description="Regression splines",
    url="https://github.com/mvds314/regspline",
    author="Martin van der Schans",
    license="BSD",
    keywords="statistics",
    packages=["regspline"],
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
    ],
    extras_require={"SVR": ["scikit-learn"], "LASSO": ["cvxopt!=1.3.0"], "FASTQR": ["pyqreg"]}
    # Note there is a domain error bug in 1.3.0: https://github.com/cvxopt/cvxopt/issues/202
)
