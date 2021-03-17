#from . import ePIE_reconstructor, mPIE_reconstructor, pSD_reconstructor
# engines available by default
from .ePIE_reconstructor import ePIE
from .pSD_reconstructor import pSD
from .mPIE_reconstructor import mPIE
from .e3PIE_reconstructor import e3PIE
from .mqNewton_reconstructor import mqNewton
from .aPIE import aPIE
from .multiPIE_reconstructor import multiPIE
from .zPIE_reconstructor import zPIE
# for other engines (like one you are developing but which is too specific) you can always import fracPy.engines.<your_engine_filename>.<your_class>
from .BaseReconstructor import BaseReconstructor


engine_dict = {'SD': pSD_reconstructor.pSD,
              'ePIE': ePIE_reconstructor.ePIE,
              'mPIE': mPIE_reconstructor.mPIE,
              }
