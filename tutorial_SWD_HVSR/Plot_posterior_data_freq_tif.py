# Plot posterior data in frequency
import os
import glob
import logging
from turtle import st
import numpy as np
import os.path as op
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib
matplotlib.use('PDF')
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)

from BayHunter import utils
from BayHunter import Targets
from BayHunter import Model, ModelMatrix
from BayHunter import SynthObs
import matplotlib.colors as colors

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rstate = np.random.RandomState(333)

plt.rcParams.update({'font.size': 20})

# Load true data
station = 'shallow4'
x_sw, _y_sw = np.loadtxt('observed/'+station+'_rdispph.dat').T
x_swg, _y_swg = np.loadtxt('observed/'+station+'_rdispgr.dat').T
#xrf, _yrf = np.loadtxt('observed/'+station+'_prf.dat').T
x_rwe_p, _y_rwe_p = np.loadtxt('observed/'+station+'_rwe.dat').T
x_rwe = 1/x_rwe_p
_y_rwe = _y_rwe_p

def brocher1(vp):
    #     Brocher BSSA 2005, EQ 1
#     1.5 km/s< Vp < 8.5 km/s
#     returns rho in g/cm^3
    vpsq = vp*vp
    vpsq2 = vpsq*vpsq
    rho = 1.6612*vp-0.4721*vpsq+0.0671*vpsq*vp-0.0043*vpsq2+0.000106*vpsq2*vp
    return rho

def brocher2(vs, vp):
#     DM Boore; 
#     http://www.daveboore.com/daves_notes/daves_notes_on_relating_density_to_velocity_v3.0.pdf
#     0 < Vs < 0.3 km/s
#     returns rho in g/cm^3
    if vs<0.3:
        rho = 1. + (1.53*vs**0.85)/(0.35 + 1.889*vs**1.7)
    else:
        rho = 1.74*vp**0.25
    return rho

class PlotPosteriorData(object):
    """
    Plot posterior data.

    """
    def __init__(self, configfile):
        condict = self.read_config(configfile)
        self.targets = condict['targets']
        self.ntargets = len(self.targets)
        self.refs = condict['targetrefs'] + ['joint']
        self.priors = condict['priors']
        self.initparams = condict['initparams']

        self.datapath = op.dirname(configfile)
        self.figpath = self.datapath.replace('data', '')
        print('Current data path: %s' % self.datapath)

        self.init_filelists()
        self.init_outlierlist()

        self.mantle = self.priors.get('mantle', None)

        self.refmodel = {'model': None,
                         'nlays': None,
                         'noise': None,
                         'vpvs': None}

    def read_config(self, configfile):
        return utils.read_config(configfile)

    def init_outlierlist(self):
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            self.outliers = np.loadtxt(outlierfile, usecols=[0], dtype=int)
            print('Outlier chains from file: %d' % self.outliers.size)
        else:
            print('Outlier chains from file: None')
            self.outliers = np.zeros(0)

    def init_filelists(self):
        filetypes = ['models', 'likes', 'misfits', 'noise', 'vpvs', 'ph']
        filepattern = op.join(self.datapath, 'c???_p%d%s.npy')
        files = []
        size = []

        for ftype in filetypes:
            p1files = sorted(glob.glob(filepattern % (1, ftype)))
            p2files = sorted(glob.glob(filepattern % (2, ftype)))
            files.append([p1files, p2files])
            size.append(len(p1files) + len(p2files))

        if len(set(size)) == 1:
            self.modfiles, self.likefiles, self.misfiles, self.noisefiles, \
                self.vpvsfiles, self.phfiles = files
        else:
            logger.info('You are missing files. Please check ' +
                        '"%s" for completeness.' % self.datapath)
            logger.info('(filetype, number): ' + str(zip(filetypes, size)))

    def _get_posterior_data(self, data, final, chainidx=0):
        if final:
            filetempl = op.join(self.datapath, 'c_%s.npy')
        else:
            filetempl = op.join(self.datapath, 'c%.3d_p2%s.npy' % (chainidx, '%s'))

        outarrays = []
        for dataset in data:
            datafile = filetempl % dataset
            p2data = np.load(datafile)
            outarrays.append(p2data)

        return outarrays
        # yobs = target.obsdata.y
        # misfit = target.valuation.get_rms(yobs, ymod)
        # jmisfit += misfit

    def savefig(self, fig, filename):
        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight")
            plt.close('all')

    def plot_posterior_ph(self, final=True, chainidx=0):
        targets = Targets.JointTarget(targets=self.targets)
        models, = self._get_posterior_data(['models'], final, chainidx)
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        period = targets.targets[0].obsdata.x  # 

        interp = period[1] - period[0]
        period_int = np.arange(period[0], period[-1] + interp / 20., interp / 20.)
        periodbins = np.arange(period[0], period[-1] + interp/10, interp/10)  # interp period bins

        periods_int = np.repeat([period_int], len(models), axis=0)
        phs_int = np.empty((len(models), period_int.size))

        for i in range(len(models)):
            # for period, phase 2D histogram
            vp, vs, h = Model.get_vp_vs_h(models[i], vpvs[i], self.mantle)
            # rho = vp * 0.32 + 0.77
            rho = np.ones(len(vs))
            for ii in range(len(rho)):
               if vp[ii] > 1.5:
                   rho[ii] = brocher1(vp[ii])
               else:
                   rho[ii] = brocher2(vs[ii], vp[ii])

            xmod, ymod = targets.targets[0].moddata.plugin.run_model(h=h, vp=vp, vs=vs, rho=rho)
            ph_int = np.interp(period_int, xmod, ymod)
            phs_int[i,:] = ph_int

        phs_flatten = phs_int.flatten()

        phinterval = 0.0025
        ph_histmin = phs_flatten.min()-2*phinterval
        ph_histmax = phs_flatten.max()+3*phinterval
        phbins = np.arange(ph_histmin, ph_histmax, phinterval) # some buffer

        fig, axes = plt.subplots(figsize=(5, 6.5))
        data2d, xedges, yedges = np.histogram2d(periods_int.flatten(),phs_flatten, 
                                				bins=(periodbins, phbins))
        X, Y = np.meshgrid(xedges, yedges)

        #for n in range(data2d.shape[0]):
        #    maxdata = np.nanmax(data2d[n,:])
        #    if np.nanmax(data2d[n,:])==0:
        #        continue
        #    data2d[n,:] = data2d[n,:]/maxdata

        pc = axes.pcolormesh(X, Y, data2d.T/max(data2d.flatten()), cmap = 'jet', shading='auto')
        #pc = axes.pcolormesh(X, Y, data2d.T, norm=colors.Normalize(0, 1), cmap = 'jet', shading='auto')
        # axes.imshow(data2d.T, interpolation='nearest', extent=(xedges[0], xedges[-1],
        #                 yedges[0], yedges[-1]),
        #                 origin='lower', aspect='auto')

        axes.plot(targets.targets[0].obsdata.x, targets.targets[0].obsdata.y, 'o', color='w', markersize=2)
        axes.plot(x_sw, _y_sw, color='w', linewidth=1)

        axes.set_ylabel('Phase velocity (km/s)')
        axes.set_xlabel('Period (s)')

        #axes.set_title('%d models' % int(len(models)))
        cbar = fig.colorbar(pc, ax=axes)
        cbar.set_label('Probability')
        return fig  

    def plot_posterior_gr(self, final=True, chainidx=0):
        targets = Targets.JointTarget(targets=self.targets)
        models, = self._get_posterior_data(['models'], final, chainidx)
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        period = targets.targets[1].obsdata.x  # 

        interp = period[1] - period[0]
        period_int = np.arange(period[0], period[-1] + interp / 20., interp / 20.)
        periodbins = np.arange(period[0], period[-1] + interp/10, interp/10)  # interp period bins

        periods_int = np.repeat([period_int], len(models), axis=0)
        phs_int = np.empty((len(models), period_int.size))

        for i in range(len(models)):
            # for period, phase 2D histogram
            vp, vs, h = Model.get_vp_vs_h(models[i], vpvs[i], self.mantle)
            # rho = vp * 0.32 + 0.77
            rho = np.ones(len(vs))
            for ii in range(len(rho)):
               if vp[ii] > 1.5:
                   rho[ii] = brocher1(vp[ii])
               else:
                   rho[ii] = brocher2(vs[ii], vp[ii])

            xmod, ymod = targets.targets[1].moddata.plugin.run_model(h=h, vp=vp, vs=vs, rho=rho)
            ph_int = np.interp(period_int, xmod, ymod)
            phs_int[i,:] = ph_int

        phs_flatten = phs_int.flatten()

        phinterval = 0.0025
        ph_histmin = phs_flatten.min()-2*phinterval
        ph_histmax = phs_flatten.max()+3*phinterval
        phbins = np.arange(ph_histmin, ph_histmax, phinterval) # some buffer

        fig, axes = plt.subplots(figsize=(5, 6.5))
        data2d, xedges, yedges = np.histogram2d(periods_int.flatten(),phs_flatten,
                                                                bins=(periodbins, phbins))
        X, Y = np.meshgrid(xedges, yedges)

        #for n in range(data2d.shape[0]):
        #    maxdata = np.nanmax(data2d[n,:])
        #    if np.nanmax(data2d[n,:])==0:
        #        continue
        #    data2d[n,:] = data2d[n,:]/maxdata


        pc = axes.pcolormesh(X, Y, data2d.T/max(data2d.flatten()), cmap = 'jet', shading='auto')
        # axes.imshow(data2d.T, interpolation='nearest', extent=(xedges[0], xedges[-1],
        #                 yedges[0], yedges[-1]),
        #                 origin='lower', aspect='auto')

        axes.plot(targets.targets[1].obsdata.x, targets.targets[1].obsdata.y, 'o', color='w', markersize=2)
        axes.plot(x_swg, _y_swg, color='w', linewidth=1)

        axes.set_ylabel('Group velocity (km/s)')
        axes.set_xlabel('Period (s)')

        #axes.set_title('%d models' % int(len(models)))
        cbar = fig.colorbar(pc, ax=axes)
        cbar.set_label('Probability')
        return fig


    def plot_posterior_rf(self, final=True, chainidx=0):
        targets = Targets.JointTarget(targets=self.targets)
        models, = self._get_posterior_data(['models'], final, chainidx)
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        period = targets.targets[1].obsdata.x  # 

        interp = period[1] - period[0]
        period_int = np.arange(period[0], period[-1] + interp / 2., interp / 2.)
        periodbins = np.arange(period[0], int(period[-1]) + interp, interp)  # interp period bins

        periods_int = np.repeat([period_int], len(models), axis=0)
        rfs_int = np.empty((len(models), period_int.size))

        for i in range(len(models)):
            # for period, phase 2D histogram
            vp, vs, h = Model.get_vp_vs_h(models[i], vpvs[i], self.mantle)
            rho = vp * 0.32 + 0.77
            xmod, ymod = targets.targets[1].moddata.plugin.run_model(h=h, vp=vp, vs=vs, rho=rho)
            rf_int = np.interp(period_int, xmod, ymod)
            rfs_int[i,:] = rf_int

        phs_flatten = rfs_int.flatten()

        phinterval = 0.002
        ph_histmin = phs_flatten.min()-2*phinterval
        ph_histmax = phs_flatten.max()+3*phinterval
        phbins = np.arange(ph_histmin, ph_histmax, phinterval) # some buffer

        fig, axes = plt.subplots(figsize=(5, 6.5))
        data2d, xedges, yedges = np.histogram2d(periods_int.flatten(),phs_flatten, 
                                				bins=(periodbins, phbins))
        X, Y = np.meshgrid(xedges, yedges)

        pc = axes.pcolormesh(X, Y, data2d.T, norm=colors.Normalize(0, 1), cmap = 'jet', shading='nearest')
        #pc = axes.pcolormesh(X, Y, data2d.T/max(data2d.flatten()), cmap = 'jet', shading='auto')
        # axes.imshow(data2d.T, interpolation='nearest', extent=(xedges[0], xedges[-1],
        #                 yedges[0], yedges[-1]),
        #                 origin='lower', aspect='auto')

        axes.plot(targets.targets[1].obsdata.x, targets.targets[1].obsdata.y, 'r')

        axes.set_ylabel('Amplitude')
        axes.set_xlabel('Times (s)')

        axes.set_title('%d models' % int(len(models)))
        cbar = fig.colorbar(pc, ax=axes)
        cbar.set_label('Probability')
        return fig

    def plot_posterior_rwe(self, final=True, chainidx=0):
        targets = Targets.JointTarget(targets=self.targets)
        models, = self._get_posterior_data(['models'], final, chainidx)
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        period = 1/targets.targets[2].obsdata.x[::-1]  # 

        interp = period[1] - period[0]
        period_int = np.arange(period[0], period[-1] + interp / 2., interp /2.)
        periodbins = np.arange(period[0], period[-1] + interp*2, interp)  # interp period bins
        # periodbins=periodbins[::-1]

        periods_int = np.repeat([period_int], len(models), axis=0)
        phs_int = np.empty((len(models), period_int.size))

        for i in range(len(models)):
            # for period, phase 2D histogram
            vp, vs, h = Model.get_vp_vs_h(models[i], vpvs[i], self.mantle)
            rho = np.ones(len(vs))
            for ii in range(len(rho)):
                if vp[ii] > 1.5:
                    rho[ii] = brocher1(vp[ii])
                else:
                    rho[ii] = brocher2(vs[ii], vp[ii])

            xmod, ymod = targets.targets[2].moddata.plugin.run_model(h=h, vp=vp, vs=vs, rho=rho)
            ph_int = np.interp(period_int, 1/xmod[::-1], ymod[::-1])
            phs_int[i,:] = ph_int
        print(1)
        phs_flatten = phs_int.flatten()

        phinterval = 0.004
        ph_histmin = phs_flatten.min()-1*phinterval
        ph_histmax = targets.targets[2].obsdata.y.max()+2*phinterval
        phbins = np.arange(ph_histmin, ph_histmax, phinterval) # some buffer

        fig, axes = plt.subplots(figsize=(5, 6.5))
        print(2)
        data2d, xedges, yedges = np.histogram2d(periods_int.flatten(),phs_flatten, 
                                				bins=(periodbins, phbins))
        X, Y = np.meshgrid(xedges, yedges)
        print(3)
        axes.semilogx(1/targets.targets[2].obsdata.x, targets.targets[2].obsdata.y, 'o', color='w', markersize=2)
        axes.semilogx(x_rwe, _y_rwe, color='w', linewidth=1)
        print(4)

        #for n in range(data2d.shape[0]):
        #    maxdata = np.nanmax(data2d[n,:])
        #    if np.nanmax(data2d[n,:])==0:
        #        continue
        #    data2d[n,:] = data2d[n,:]/maxdata

        pc = axes.pcolormesh(X, Y, data2d.T/max(data2d.flatten()), cmap = 'jet', shading='auto')
        # axes.imshow(data2d.T, interpolation='nearest', extent=(xedges[0], xedges[-1],
        #                 yedges[0], yedges[-1]),
        #                 origin='lower', aspect='auto')

        # axes.plot(targets.targets[1].obsdata.x, targets.targets[1].obsdata.y, 'o', color='r', markersize=2)
        axes.set_ylim([min(Y.flatten()), max(Y.flatten())])
        axes.set_ylabel('HVSR')
        axes.set_xlabel('Frequency (Hz)')

        #axes.set_title('%d models' % int(len(models)))
        cbar = fig.colorbar(pc, ax=axes)
        cbar.set_label('Probability')
        return fig

    def save_plots(self):
        fig2a = self.plot_posterior_ph()
        self.savefig(fig2a, 'c_posterior_ph.tif')

        fig2b = self.plot_posterior_gr()
        self.savefig(fig2b, 'c_posterior_gr.tif')

        fig2c = self.plot_posterior_rwe()
        self.savefig(fig2c, 'c_posterior_rwe.tif')

station = 'shallow4'
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)

# Load observed data (synthetic test data)
xsw, ysw = np.loadtxt('observed/'+station+'_rdispph.dat').T
xswg, yswg = np.loadtxt('observed/'+station+'_rdispgr.dat').T
xrwe, yrwe = np.loadtxt('observed/'+station+'_rwe.dat').T

# add noise to create observed data
# order of noise values (correlation, amplitude):
# noise = [corr1, sigma1, corr2, sigma2] for 2 targets
#noise = [0.0, 0.012, 0.98, 0.005, 0.0, 0.02]
#ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise[0], sigma=noise[1])
#ysw = _ysw + ysw_err
#yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
#yrf = _yrf + yrf_err
#yrwe_err = SynthObs.compute_expnoise(_yrwe, corr=noise[4], sigma=noise[5])
#yrwe = _yrwe + yrwe_err


#
# -------------------------------------------  get reference model for BayWatch
#
# Create truemodel only if you wish to have reference values in plots
# and BayWatch. You ONLY need to assign the values in truemodel that you
# wish to have visible.
#dep, vs = np.loadtxt('observed/st3_mod.dat', usecols=[0, 2], skiprows=1).T
#pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
#pvs = np.repeat(vs, 2)

#truenoise = np.concatenate(([noise[0]], [np.std(ysw_err)],   # target 1
#                            [noise[2]], [np.std(yrf_err)],   # target 2
#                            [noise[3]], [np.std(yrwe_err)]))  # target 3

#explike = SynthObs.compute_explike(yobss=[ysw, yrf, yrwe], ymods=[_ysw, _yrf, _yrwe],
#                                   noise=truenoise, gauss=[False, True, False],
#                                   rcond=initparams['rcond'])
#truemodel = {'model': (pdep, pvs),
#             'nlays': 3,
#             'noise': truenoise,
#             'explike': explike,
#             }

#print(truenoise, explike)


#
#  -----------------------------------------------------------  DEFINE TARGETS
#
# Only pass x and y observed data to the Targets object which is matching
# the data type. You can chose for SWD any combination of Rayleigh, Love, group
# and phase velocity. Default is the fundamendal mode, but this can be updated.
# For RF chose P or S. You can also use user defined targets or replace the
# forward modeling plugin wih your own module.
target1 = Targets.RayleighDispersionPhase(xsw, ysw)
target2 = Targets.RayleighDispersionGroup(xswg, yswg)
target3 = Targets.RayleighWaveEllipticity(xrwe, yrwe)
# Join the targets. targets must be a list instance with all targets
# you want to use for MCMC Bayesian inversion.
targets = Targets.JointTarget(targets=[target1,target2,target3])


#
#  ---------------------------------------------------  Quick parameter update
#
# "priors" and "initparams" from config.ini are python dictionaries. You could
# also simply define the dictionaries directly in the script, if you don't want
# to use a config.ini file. Or update the dictionaries as follows, e.g. if you
# have station specific values, etc.
# See docs/bayhunter.pdf for explanation of parameters

#priors.update({'mohoest': (38, 4),  # optional, moho estimate (mean, std)
#               'rfnoise_corr': 0.98,
#               'swdnoise_corr': 0.,
#               'rwenoise_corr': 0.
               # 'rfnoise_sigma': np.std(yrf_err),  # fixed to true value
               # 'swdnoise_sigma': np.std(ysw_err),  # fixed to true value
#               })

initparams.update({'nchains': 96,
                   'iter_burnin': (1000000),
                   'iter_main': (500000),
                   'propdist': (0.025, 0.025, 0.015, 0.005, 0.005),
                   })
path = initparams['savepath']
cfile = '%s_config.pkl' % initparams['station']
configfile = op.join(path, 'data', cfile)
obj = PlotPosteriorData(configfile)
obj.save_plots()
