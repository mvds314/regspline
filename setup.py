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
    install_requires=["numpy", "pandas", "scikit-learn", "statsmodels", "cvxopt"],
)
