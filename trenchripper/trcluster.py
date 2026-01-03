# fmt: off
import os
import shutil
import dask
import time
import h5py
import threading
import dask.distributed

from time import sleep
from dask.distributed import Client,progress
from dask.distributed.diagnostics.plugin import WorkerPlugin
from dask_jobqueue import SLURMCluster
from IPython.display import display, HTML
from .utils import writedir

## Hacky memory trim
import ctypes
import gc

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

class dask_controller: #adapted from Charles' code
    def __init__(self,n_workers=6,n_workers_min=6,local=True,queue="short",death_timeout=60.,\
                 walltime='01:00:00',cores=1,processes=1,memory='6GB',\
                 working_directory="./",job_extra_directives=[],account=None):
        self.local = local
        self.n_workers = n_workers
        self.n_workers_min = n_workers_min
        self.walltime = walltime
        self.queue = queue
        self.death_timeout = death_timeout
        self.processes = processes
        self.memory = memory
        self.cores = cores
        self.working_directory = working_directory
        self.job_extra_directives = job_extra_directives
        self.account = account

        self.futures = {}

        split_walltime = walltime.split(":")
        wall_hrs,wall_mins,wall_secs = tuple(split_walltime)
        ttl_wall_mins = (60*int(wall_hrs)) + int(wall_mins)

        if ttl_wall_mins < 10:
            raise ValueError("Walltime must be at least 10 mins long!")

        self.lifetime = str(max(ttl_wall_mins-10,10)) + "m"
        print(self.lifetime)
        print(walltime)
        self.worker_extra_args = ["--lifetime", self.lifetime, "--lifetime-stagger", "5m"]

        writedir(working_directory,overwrite=False)

    def startdask(self):
        if self.local:
            self.daskclient = Client()
            self.daskclient.cluster.scale(self.n_workers)
        else:
            if self.account is None:
                self.daskcluster = SLURMCluster(n_workers=self.n_workers_min,queue=self.queue,death_timeout=self.death_timeout,walltime=self.walltime,\
                                       processes=self.processes,memory=self.memory,cores=self.cores,local_directory=self.working_directory,\
                                    log_directory=self.working_directory,worker_extra_args=self.worker_extra_args,job_extra_directives=self.job_extra_directives)
            else:
                self.daskcluster = SLURMCluster(n_workers=self.n_workers_min,queue=self.queue,death_timeout=self.death_timeout,walltime=self.walltime,\
                                       processes=self.processes,memory=self.memory,cores=self.cores,local_directory=self.working_directory,\
                                    log_directory=self.working_directory,worker_extra_args=self.worker_extra_args,job_extra_directives=self.job_extra_directives,account=self.account)
            self.daskcluster.adapt(minimum=self.n_workers_min, maximum=self.n_workers,\
                                   interval="1m",wait_count=10)
            self.daskclient = Client(self.daskcluster)

    def shutdown(self, delete_files=True):
        self.reset_worker_memory()
        if not self.local:
            self.daskcluster.close()
        if delete_files:
            for item in os.listdir(self.working_directory):
                if "worker-" in item or "slurm-" in item or ".lock" in item:
                    path = "./" + item
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)

    def printprogress(self):
        complete = len([item for item in self.futures if item.status=="finished"])
        print(str(complete) + "/" + str(len(self.futures)))

    def displaydashboard(self):
        link = self.daskcluster.dashboard_link
        display(HTML('<a href="' + link +'">Dashboard</a>'))

    def mapfovs(self,function,fov_list,retries=0):
        self.function = function
        self.retries = retries
        def mapallfovs(fov_number,function=function):
            function(fov_number)
        self.futures = {}
        for fov in fov_list:
            future = self.daskclient.submit(mapallfovs,fov,retries=retries)
            self.futures[fov] = future

    def retry_failed(self):
        self.failed_fovs = [fov for fov,future in self.futures.items() if future.status != 'finished']
        out = self.daskclient.restart()
        self.mapfovs(self.function,self.failed_fovs,retries=self.retries)

    def retry_processing(self):
        self.proc_fovs = [fov for fov,future in self.futures.items() if future.status == 'pending']
        out = self.daskclient.restart()
        self.mapfovs(self.function,self.proc_fovs,retries=self.retries)

    def restart(self):
        self.daskclient.restart()
        with self.failure_lock:
            self.failure_counter_variable.set(0)

    def reset_worker_memory(self):
        self.daskclient.cancel([val for key,val in self.futures.items()])
        self.daskclient.run(gc.collect)
        self.daskclient.run(trim_memory)
        print("Done.")

class hdf5lock:
    def __init__(self,filepath,updateperiod=0.1):
        self.filepath = filepath
        self.lockfile = filepath + ".lock"
        self.updateperiod = updateperiod

    def _lock(self):
        while True:
            if not os.path.exists(self.lockfile):
                open(self.lockfile,'w').close()
                break
            sleep(self.updateperiod)

    def _apply_fn(self,function,iomode,*args,**kwargs):
        try:
            fn_output = function(self.filepath,iomode,*args,**kwargs)
            os.remove(self.lockfile)
            return fn_output
        except:
            os.remove(self.lockfile)
            raise

    def lockedfn(self,function,iomode,*args,**kwargs):
        self._lock()
        fn_output = self._apply_fn(function,iomode,*args,**kwargs)
        return fn_output

def transferjob(sourcedir,targetdir,single_file=False):
    mkdircmd = "mkdir -p '" + targetdir + "'"
    if single_file:
        rsynccmd = "rsync -r '" + sourcedir + "' '" + targetdir + "'"
    else:
        rsynccmd = "rsync -r '" + sourcedir + "/' '" + targetdir + "'"
    wrapcmd = mkdircmd + " && " + rsynccmd
    cmd = "sbatch -p transfer -t 0-12:00 --wrap=\"" + wrapcmd + "\""
    print(cmd)
    os.system(cmd)
