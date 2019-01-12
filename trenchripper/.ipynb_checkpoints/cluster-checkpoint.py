import trenchripper as tr
import os
import shutil
import dask

from dask.distributed import Client,progress
from dask_jobqueue import SLURMCluster

class dask_controller: #adapted from Charles' code
    def __init__(self,n_workers=6,local=True,queue="short",\
                 walltime='01:30:00',cores=1,processes=1,memory='6GB'):
        self.local = local
        self.n_workers = n_workers
        self.walltime = walltime
        self.queue = queue
        self.processes = processes
        self.memory = memory
        self.cores = cores
        
    def writedir(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def startdask(self):
        if self.local:
            self.daskclient = Client()
            self.daskclient.cluster.scale(self.n_workers)
        else:
            self.daskcluster = SLURMCluster(queue=self.queue,walltime=self.walltime,\
                                   processes=self.processes,memory=self.memory,
                                  cores=self.cores)
            self.workers = self.daskcluster.start_workers(self.n_workers)
            self.daskclient = Client(self.daskcluster)
    
    def shutdown(self):
        self.daskcluster.stop_all_jobs()
        for item in os.listdir("./"):
            if "dask-worker" in item:
                path = "./" + item
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            
    def printprogress(self):
        complete = len([item for item in self.futures if item.status=="finished"])
        print(str(complete) + "/" + str(len(self.futures)))
        
    def mapfovs(self,function,ttl_fovs):
        fovs = list(range(ttl_fovs))
        def mapallfovs(fov_number,function=function):
            function(fov_number)
        self.futures = self.daskclient.map(mapallfovs,fovs)