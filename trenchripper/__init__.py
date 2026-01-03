import shutil
from pathlib import Path
import os
from importlib.resources import files

from .kymograph import *
from .interactive import *
from .tplot import *
from .trcluster import *
from .ndextract import *
from .segment import *

## from .unet import *
from .metrics import *
from .analysis import *
from .tracking import *
from .projection import *

# from .kymograph_template_match import *
# from .phase_tracking import *
from .marlin import *
from .daskutils import *
from .utils import *
from .steadystate import *

# jobqueue_config_path = (
#     f"{os.path.dirname(os.path.abspath(__file__))}" + "/jobqueue.yaml"
# )

jobqueue_config_path = files("trenchripper").joinpath("jobqueue.yaml")
system_jobqueue_config_path = Path.home() / ".config" / "dask" / "jobqueue.yaml"
system_jobqueue_config_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(jobqueue_config_path, system_jobqueue_config_path)
