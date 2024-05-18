import shutil
import os

from .kymograph import *
from .interactive import *
from .tplot import *
from .trcluster import *
from .ndextract import *
from .segment import *

from .metrics import *
from .analysis import *
from .tracking import *
from .projection import *
from .marlin import *
from .daskutils import *
from .utils import *
from .steadystate import *

jobqueue_config_path = (
    f"{os.path.dirname(os.path.abspath(__file__))}" + "/jobqueue.yaml"
)
system_jobqueue_config_path = os.path.expanduser("~/.config/dask/jobqueue.yaml")
shutil.copyfile(jobqueue_config_path, system_jobqueue_config_path)
