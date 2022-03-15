"""
The :mod:`teacher.explanation` module includes the methods to generate
factual and counterfactual explanations, as well as the `Explainer` classes
"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._factual import FID3_factual, m_factual, mr_factual, c_factual
from ._counterfactual import FID3_counterfactual, i_counterfactual, f_counterfactual
from .FID3_explainer import FID3Explainer
from .FDT_explainer import FDTExplainer

# =============================================================================
# Public objects
# =============================================================================

# Set the classes that are accessible
# from the module teacher.explanation
__all__ = [
    "FID3_factual",
    "m_factual",
    "mr_factual",
    "c_factual",
    "FID3_counterfactual",
    "i_counterfactual",
    "f_counterfactual",
    "FID3Explainer",
    "FDTExplainer"
]
