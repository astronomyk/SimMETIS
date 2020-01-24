"""
End-to-end simulator for METIS on the E-ELT
============================================
"""
# turn off warnings - interesting for development, but not for runtime
import warnings
import logging

from os.path import join

from astropy.utils.exceptions import AstropyWarning

# Import all the modules to go under simmetis
from . import utils

from . import spectral
from . import spatial
from . import psf

from . import detector
from . import optics
from . import commands
from . import source

from . import simulation

# spectro is not yet in source distribution
try:
    from . import spectro
except ImportError:
    pass
try:
    from .version import version as __version__
except ImportError:
    __version__ = "Version number is not available"

# import specific Classes from the modules to be accessible in the global
# namespace
from .utils     import __pkg_dir__
from .utils     import bug_report
from .detector  import Detector, Chip
from .source    import Source
from .optics    import OpticalTrain
from .commands  import UserCommands

# don't import these ones just yet
#from .SpectralGrating  import *
from .simulation import run
from .utils import get_extras
from .detector import install_noise_cube

__data_dir__ = join(__pkg_dir__, "data")

# Search path for finding files
__search_path__ = ['./', __pkg_dir__, __data_dir__]


logging.basicConfig(filename='simmetis.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.info("SimMETIS imported")

#warnings.simplefilter('ignore', UserWarning)   # user should see UserWarnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)  # warnings for the developer
warnings.simplefilter('ignore', category=AstropyWarning)
