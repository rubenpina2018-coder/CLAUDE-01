"""Card-fraud detection: data contract, featurizer and quantized GBDT runtime.

Everything importable from this package depends only on numpy and pydantic, so
the inference container does not ship scikit-learn, scipy or pandas.
"""

__version__ = "1.0.0"
