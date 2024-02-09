# __init__.py

# Import any modules or sub-packages that you want to make available when the package is imported
from .data import *
from .utils import *
from .analysis import *
from .visualization import *
from .settings import *
from .graph import *
from .raster import *

# Define any variables or constants that you want to be accessible from outside the package
__version__ = '0.0.1'

# Optionally, you can define a list of modules or sub-packages that should be imported when using the wildcard import statement
__all__ = ['analysis', 'data', 'visualization']
