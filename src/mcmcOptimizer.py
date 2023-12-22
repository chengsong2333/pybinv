# #############################
#
# Cheng Song (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling
#
#
# #############################

import os
import time
import os.path as op
import numpy as np


import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import sharedctypes
from multiprocessing import Lock

import matplotlib.cm as cm
from collections import OrderedDict

from pybinv.utils import SerializingContext
from pybinv import Model, ModelMatrix
from pybinv import SingleChain
from pybinv import utils

import logging
logger = logging.getLogger()

class MCMC_Optimizer(object):
    """
    Contains multiple chains - parallel computing.
    Check output files/ folder of forward modeling to not cause errors
    """
    def __init__(self, targets, initparams=dict(), priors=dict(),
                 random_seed=None, initmodel=None, parallel_tempering=False):
        self.sock_addr = 'tcp://*:5556'
        self.rstate = np.random.RandomState(random_seed)

        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.priors.update(priors)
        self.initparams.update(initparams)
        self.initmodel = initmodel
        self.parallel_tempering = parallel_tempering

        self.station = self.initparams.get('station')

        savepath = op.join(self.initparams['savepath'], 'data')
        if not op.exists(savepath):
            os.makedirs(savepath)

        # save file for offline-plotting
        outfile = op.join(savepath, '%s_config.pkl' % self.station)
        utils.save_config(targets, outfile, priors=self.priors,
                          initparams=self.initparams)

        self.nchains = self.initparams.get('nchains')
        self.ntargets = len(targets.targets)

        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iterations = self.iter_phase1 + self.iter_phase2

        self.maxlayers = int(self.priors['layers'][1]) + 1
        # self.vpvs = self.priors['vpvs']

        # shared data and chains
        self._init_shareddata()

        self._init_parallel_tempering(self.parallel_tempering)

        logger.info('> %d chain(s) are initiated ...' % self.nchains)

        self.chains = []
        for i in np.arange(self.nchains):
            self.chains += [self._init_chain(chainidx=i, targets=targets)]

        self.manager = mp.Manager()

    def _init_shareddata(self):
        """Create a shared raw array.

        All models / likes will be saved and load from this array.
        """
        lock=Lock()
        memory = 0
        logger.info('> Chain arrays are initiated...')
        dtype = np.float32

        acceptance = np.max(self.initparams['acceptance']) / 100.
        accepted_models = int(self.iterations * acceptance)
        self.nmodels = accepted_models  # 'iterations'

        # models
        self.sharedmodels = sharedctypes.RawArray(
            'f', self.nchains * (self.nmodels * self.maxlayers * 2))
        modeldata = np.frombuffer(self.sharedmodels, dtype=dtype)
        modeldata.fill(np.nan)
        memory += modeldata.nbytes

        # misfits array collects misfits for each target and the jointmisfit
        self.sharedmisfits = sharedctypes.RawArray(
            'f', self.nchains * self.nmodels * (self.ntargets + 1))
        misfitdata = np.frombuffer(self.sharedmisfits, dtype=dtype)
        misfitdata.fill(np.nan)
        memory += misfitdata.nbytes

        # likelihoods
        self.sharedlikes = sharedctypes.RawArray(
            'f', self.nchains * self.nmodels)
        likedata = np.frombuffer(self.sharedlikes, dtype=dtype)
        likedata.fill(np.nan)
        memory += likedata.nbytes

        # noise hyper-parameters, which are for each target two:
        # noise correlation r, noise amplitudes sigma
        self.sharednoise = sharedctypes.RawArray(
            'f', self.nchains * self.nmodels * self.ntargets*2)
        noisedata = np.frombuffer(self.sharednoise, dtype=dtype)
        noisedata.fill(np.nan)
        memory += noisedata.nbytes

        # vpvs
        self.sharedvpvs = sharedctypes.RawArray(
            'f', self.nchains * self.nmodels)
        vpvsdata = np.frombuffer(self.sharedvpvs, dtype=dtype)
        vpvsdata.fill(np.nan)
        memory += vpvsdata.nbytes

        # Shared temperature for parallel tempering
        # t1 = np.ones(int(self.nchains/4)) # select 1/4 of chains with T=1
        # T = np.concatenate((t1,np.logspace(np.log10(1),np.log10(50),self.nchains-np.size(t1)))) # T=1-50
        # self.sharedbeta = sharedctypes.Array(
        #     'f', 1/T, lock=True)


        # t1 = np.ones(int(self.nchains/4)) # select 1/4 of chains with T=1
        # T = np.concatenate((t1,np.logspace(np.log10(1),np.log10(50),self.nchains-np.size(t1)))) # T=1-50

        # temperatures (has to be as large as the total number of iterations)
        self.sharedtemperatures = sharedctypes.RawArray(
            'f', self.nchains * self.iterations)
        temperatures = np.frombuffer(self.sharedtemperatures, dtype=dtype)
        temperatures.fill(np.nan)
        memory += temperatures.nbytes


        # current likelihood (otherwise to slow to find the last
        # not nan value in lself.sharedlikes)
        self.sharedlikes_current = sharedctypes.RawArray(
            'f', self.nchains)
        likes_curr = np.frombuffer(self.sharedlikes_current,
                                   dtype=dtype)
        likes_curr.fill(np.nan)
        memory += likes_curr.nbytes

        memory = np.ceil(memory / 1e6)
        logger.info('... they occupy ~%d MB memory.' % memory)

    def _init_chain(self, chainidx, targets):
        chain = SingleChain(
            targets=targets, chainidx=chainidx, modelpriors=self.priors,
            initparams=self.initparams, sharedmodels=self.sharedmodels,
            sharedmisfits=self.sharedmisfits, sharedlikes=self.sharedlikes,
            sharednoise=self.sharednoise, sharedvpvs=self.sharedvpvs,
            sharedtemperatures=self.sharedtemperatures, sharedlikes_current=self.sharedlikes_current,
            random_seed=self.rstate.randint(1000), initmodel=self.initmodel)

        return chain

    def _init_parallel_tempering(self,tempering):
    
        sharedtemperatures = np.frombuffer(self.sharedtemperatures, dtype=np.float32).\
            reshape((self.nchains,self.iterations))

        if tempering:
            self.temperatures = sharedtemperatures
            t1 = np.ones(int(self.nchains/4)) # select 1/4 of chains with T=1
            temperatures = np.concatenate((t1,np.logspace(np.log10(1),np.log10(2),self.nchains-np.size(t1)))) # T=1-2
            self.temperatures[:,0] = temperatures
            self.likelihoods = np.frombuffer(self.sharedlikes_current,
                                             dtype=np.float32)
            self.accepted_temperature_swaps = 0
            self.total_temperature_swaps = 0
            self.accrate_temp_swaps = np.zeros(100)
        else:
            sharedtemperatures[:,:] = 1.

    def _swap_temperatures(self,iiter):

        temps = self.temperatures[:,iiter-1]
        likes = self.likelihoods

        # temperatures at the next iteration are the same as current
        # unless swap is accepted (below)
        self.temperatures[:,iiter] = temps

        chains = np.arange(self.nchains,dtype=int)
        self.rstate.shuffle(chains)
        
        # swap between random pairs of chains
        for pair in chains[:int(len(chains)/2.)*2].reshape(int(len(chains)/2),2):

            # no swap if temperatures are identical
            if temps[pair[0]] == temps[pair[1]]:
                continue

            # check if the two temperature levels are adjacent
            if np.diff(np.where(np.isin(np.unique(temps),[temps[pair[0]],temps[pair[1]]]))[0])[0] == 1:
                adjacent = True
                self.accrate_temp_swaps[0] = 0
            else:
                adjacent = False

            u = np.log(self.rstate.uniform(0, 1))
            alpha = (likes[pair[1]]-likes[pair[0]]) * (1./temps[pair[0]] - 1./temps[pair[1]])
            if u <= alpha:
                self.temperatures[pair[0],iiter] = temps[pair[1]]
                self.temperatures[pair[1],iiter] = temps[pair[0]]
                if adjacent:
                    self.accrate_temp_swaps[0] = 1
                self.accepted_temperature_swaps += 1
            self.total_temperature_swaps += 1
            self.accrate_temp_swaps = np.roll(self.accrate_temp_swaps,1)

    def monitor_process(self, dtsend):
        """Create a socket and send array data. Only active for baywatch."""
        import zmq
        context = SerializingContext()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(self.sock_addr)
        dtype = np.float32

        logger.info('Starting monitor process on %s...' % self.sock_addr)

        models = np.frombuffer(self.sharedmodels, dtype=dtype) \
            .reshape((self.nchains, self.nmodels, self.maxlayers*2))
        likes = np.frombuffer(self.sharedlikes, dtype=dtype) \
            .reshape((self.nchains, self.nmodels))
        noise = np.frombuffer(self.sharednoise, dtype=dtype) \
            .reshape((self.nchains, self.nmodels, self.ntargets*2))
        vpvs = np.frombuffer(self.sharedvpvs, dtype=dtype) \
            .reshape((self.nchains, self.nmodels))

        def get_latest_row(models):
            nan_mask = ~np.isnan(models[:, :, 0])
            model_mask = np.argmax(np.cumsum(nan_mask, axis=1), axis=1)
            latest_models = [models[ic, model_mask[ic], :]
                             for ic in range(self.nchains)]
            return np.vstack(latest_models)

        def get_latest_likes(likes):
            nan_mask = ~np.isnan(likes[:, :])
            like_mask = np.argmax(np.cumsum(nan_mask, axis=1), axis=1)
            latest_likes = [likes[ic, like_mask[ic]]
                            for ic in range(self.nchains)]
            return np.vstack(latest_likes)

        def get_latest_noise(noise):
            nan_mask = ~np.isnan(models[:, :, 0])
            noise_mask = np.argmax(np.cumsum(nan_mask, axis=1), axis=1)
            latest_noise = [noise[ic, noise_mask[ic], :]
                            for ic in range(self.nchains)]
            return np.vstack(latest_noise)

        def get_latest_vpvs(vpvs):
            nan_mask = ~np.isnan(vpvs[:, :])
            vpvs_mask = np.argmax(np.cumsum(nan_mask, axis=1), axis=1)
            latest_vpvs = [vpvs[ic, vpvs_mask[ic]]
                           for ic in range(self.nchains)]
            return np.vstack(latest_vpvs)

        while True:
            logger.debug('Sending array...')
            latest_models = get_latest_row(models)
            latest_likes = get_latest_likes(likes)
            latest_noise = get_latest_noise(noise)
            latest_vpvs = get_latest_vpvs(vpvs)

            latest_vpvs_models = \
                np.concatenate((latest_vpvs, latest_models), axis=1)

            self.socket.send_array(latest_vpvs_models)
            self.socket.send_array(latest_likes)
            self.socket.send_array(latest_noise)
            time.sleep(dtsend)


    def mp_inversion(self, baywatch=False, dtsend=0.5, nthreads=0):
        """Multiprocessing inversion."""

        def idxsort(chain):
            return chain.chainidx

        def gochain(threadno,worklist,barrier):

            iiter = 0
            while True:
                chains_done = 0
                for chainidx in worklist:
                    chain = self.chains[chainidx]
                    if chain.iiter < chain.iter_phase2:
                        chain.iterate()
                    else:
                        chains_done += 1

                iiter += 1

                if chains_done == len(worklist):
                    break

                if self.parallel_tempering and iiter<self.iterations:
                    barrier.wait() # wait for all chains
                    if threadno==1:
                        self._swap_temperatures(iiter)
                        if iiter%5000 == 0:
                            logger.info("Accepted temperature swaps: %d / %d (Acceptance rate adjacent temp. levels: %d%%)" %(
                                self.accepted_temperature_swaps, self.total_temperature_swaps,
                                np.sum(self.accrate_temp_swaps)))
                    # make all threads wait until swap is finished
                    barrier.wait()

            for chainidx in worklist:
                chain = self.chains[chainidx]
                chain.finalize()
            
                # reset to None, otherwise pickling error
                for target in chain.targets.targets:
                    target.get_covariance = None

            self.chainlist.append(chain)

        if baywatch:
            baywatch_thread=1
        else:
            baywatch_thread=0
        # multi processing - parallel chains
        if nthreads == 0:
            nthreads = mp.cpu_count()
        # if self.nchains+baywatch_thread <= nthreads:
        #     nthreads = self.nchains+baywatch_thread
        #     logger.info("Reducing number of threads to %d." %nthreads)
            
        # distribute the chains equally among worker threads
        # if baywatch, use one thread for baywatch process
        # worklist = [list(np.arange(self.nchains)[i::nthreads-baywatch_thread]) for i in range(nthreads-baywatch_thread)]
        worklist = [list(np.arange(self.nchains)[i::nthreads-baywatch_thread]) for i in range(nthreads-baywatch_thread)]

        self.chainlist = self.manager.list()
        self.alive = []
        t0 = time.time()

        if baywatch:
            monitor = mp.Process(
                name='BayWatch',
                target=self.monitor_process,
                kwargs={'dtsend': dtsend})
            monitor.start()
        
        barrier = mp.Barrier(nthreads)
        for i,chains in enumerate(worklist):
            thread_no = i+1
            logger.info('> Thread %d working on chain(s) ' %(thread_no) + 
                        ' '.join([str(ch) for ch in chains]))
            p = mp.Process(name='Thread %d' %thread_no, target=gochain,
                           kwargs={'threadno': thread_no, 'worklist': chains,'barrier': barrier})
            self.alive.append(p)
            p.start()

        logger.info('Chain No. (temp.): ' +
                    'iteration  layers  RMS misfit  ' +
                    'likelihood | duration | acceptance rates (total)')

        # wait for chains to finish
        while True:
            alive = [process for process in self.alive
                     if process.is_alive()]
            # all processes terminated
            if len(alive) == 0:
                if baywatch:
                    # wait for BayWatch to recognize that inversion has finished
                    time.sleep(5*dtsend)
                    monitor.terminate()
                break
            time.sleep(.5)

        p.join()

        logger.info('> All chains terminated after: %.5f s' % (time.time() - t0))

        try:
            # only necessary, if want to access chain data after an inversion,
            # i.e. all models can be accessed in the python terminal, e.g.
            # for testing purposes. This does not work, if too much memory
            # is already occupied.
            self.chains = list(self.chainlist)
            self.chains.sort(key=idxsort)
        except:
            pass

        runtime = (time.time() - t0)
        logger.info('### time for inversion: %.2f s' % runtime)