# =============================================================================
# Constants
# =============================================================================

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440
#
# Generic release markers:
#   * X.Y
#   * X.Y.Z   # Bug-fix release
#
# Admissible pre-release markers:
#   * X.YaN   # Alpha release
#   * X.YbN   # Beta release
#   * X.YrcN  # Release candidate
#   * X.Y     # Final release
#
# Dev branch marker is: "X.Y.dev" or "X.Y.devN", where N is an integer.
# "X.Y.dev0" is the canonical version of "X.Y.dev".

# Teacher package version
__version__ = "1.0.1"


# =============================================================================
# Module public objects
# =============================================================================

__all__ = {
    "datasets",
    "explanation",
    "neighbors",
    "tree",
    "utils"
}
