# zoom in Plot posterior data
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

def vs_round(vs):
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor

def tryexcept(func):
    def wrapper_tryexcept(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            print('* %s: Plotting was not possible\nErrorMessage: %s'
                  % (func.__name__, e))
            return None
    return wrapper_tryexcept

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

    def plot_refmodel(self, fig, mtype='model', **kwargs):
        if fig is not None and self.refmodel[mtype] is not None:
            if mtype == 'nlays':
                nlays = self.refmodel[mtype]
                fig.axes[0].axvline(nlays, color='red', lw=0.5, alpha=0.7)

            if mtype == 'model':
                dep, vs = self.refmodel['model']
                assert len(dep) == len(vs)
                fig.axes[0].plot(vs, dep, **kwargs)
                if len(fig.axes) == 2:
                    deps = np.unique(dep)
                    for d in deps:
                        fig.axes[1].axhline(d, **kwargs)

            if mtype == 'noise':
                noise = self.refmodel[mtype]
                for i in range(len(noise)):
                    fig.axes[i].axvline(
                        noise[i], color='red', lw=0.5, alpha=0.7)

            if mtype == 'vpvs':
                vpvs = self.refmodel[mtype]
                fig.axes[0].axvline(vpvs, color='red', lw=0.5, alpha=0.7)
        return fig

    @staticmethod
    def _plot_bestmodels(bestmodels, dep_int=None):
        fig, ax = plt.subplots(figsize=(3, 1.5))

        models = ['mean', 'median', 'stdminmax']
        colors = ['green', 'blue', 'black']
        ls = ['-', '--', ':']
        lw = [1, 1, 1]

        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int)

        for i, model in enumerate(models):
            vs, dep = singlemodels[model]

            ax.plot(vs.T, dep, color=colors[i], label=model,
                    ls=ls[i], lw=lw[i])

        ax.invert_yaxis()
        #ax.set_ylabel('Depth in km')
        #ax.set_xlabel('$V_S$ in km/s')

        han, lab = ax.get_legend_handles_labels()
        #ax.legend(han[:-1], lab[:-1], loc=3, fontsize=10)
        return fig, ax

    @staticmethod
    def _plot_bestmodels_hist(models, dep_int=None):
        """
        2D histogram with 30 vs cells and 50 depth cells.
        As plot depth is limited to 100 km, each depth cell is a 2 km.

        pinterf is the number of interfaces to be plot (derived from gradient)
        """
        if dep_int is None:
            dep_int = np.linspace(0, 100, 201)  # interppolate depth to 0.5 km.
            # bins for 2d histogram
            depbins = np.linspace(0, 100, 101)  # 1 km bins
        else:
            maxdepth = int(np.ceil(dep_int.max()))
            interp = dep_int[1] - dep_int[0]
            dep_int = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
            depbins = np.arange(0, maxdepth + 2*interp, interp)  # interp km bins
            # nbin = np.arange(0, maxdepth + interp, interp)  # interp km bins

        # get interfaces, #first
        models2 = ModelMatrix._replace_zvnoi_h(models)
        models2 = np.array([model[~np.isnan(model)] for model in models2])
        yinterf = np.array([np.cumsum(model[int(model.size/2):-1])
                            for model in models2])
        yinterf = np.concatenate(yinterf)

        vss_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)
        singlemodels = ModelMatrix.get_singlemodels(models, dep_int=depbins)

        vss_flatten = vss_int.flatten()
        vsinterval = 0.025  # km/s, 0.025 is assumption for vs_round
        # vsbins = int((vss_flatten.max() - vss_flatten.min()) / vsinterval)
        vs_histmin = vs_round(vss_flatten.min())-2*vsinterval
        vs_histmax = vs_round(vss_flatten.max())+3*vsinterval
        vsbins = np.arange(vs_histmin, vs_histmax, vsinterval) # some buffer

        # initiate plot
        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 2]},
                                 sharey=True, figsize=(4.5, 1.5))
        fig.subplots_adjust(wspace=0.05)

        data2d, xedges, yedges = np.histogram2d(vss_flatten, deps_int.flatten(),
                                                                bins=(vsbins, depbins))

        X, Y = np.meshgrid(xedges, yedges)
        pc = axes[0].pcolormesh(X, Y, data2d.T/max(data2d.flatten()), cmap = 'jet', shading='auto')

#        axes[0].imshow(data2d.T, extent=(xedges[0], xedges[-1],
#                                                                     yedges[0], yedges[-1]),
#                                   origin='lower',
#                                   vmax=len(models), aspect='auto')
       # plot mean / modes
        # colors = ['green', 'white']
        # for c, choice in enumerate(['mean', 'mode']):
        colors = ['white']
        for c, choice in enumerate(['mode']):
            vs, dep = singlemodels[choice]
            color = colors[c]
            axes[0].plot(vs, dep, color=color, lw=1, alpha=0.9, label=choice)

        vs_mode, dep_mode = singlemodels['mode']
        #axes[0].legend(loc=3, fontsize=10)

        # histogram for interfaces
        data = axes[1].hist(yinterf, bins=depbins, orientation='horizontal',
                            color='lightgray', alpha=0.7,
                            edgecolor='k')
        bins, lay_bin, _ = np.array(data).T
        center_lay = (lay_bin[:-1] + lay_bin[1:]) / 2.

        #axes[0].set_ylabel('Depth in km')
        #axes[0].set_xlabel('$V_S$ in km/s')

        axes[0].invert_yaxis()

        #axes[0].set_title('%d models' % len(models))
        axes[1].set_xticks([])
        return fig, axes

    def savefig(self, fig, filename):
        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight")
            plt.close('all')

    def plot_posterior_models1d(self, final=True, chainidx=0, depint=1):
        """depint is the depth interpolation used for binning. Default=1km."""
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        fig, ax = self._plot_bestmodels(models, dep_int)
        ax.set_xlim([0,2])
        ax.set_ylim([0.05,0])
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        #ax.set_title('%d models from %d chains' % (len(models), nchains))
        return fig

    #@tryexcept
    def plot_posterior_models2d(self, final=True, chainidx=0, depint=1):
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)

        fig, axes = self._plot_bestmodels_hist(models, dep_int)
        axes[0].set_xlim([0.25,2])
        axes[0].set_ylim([0.05,0])
        #axes[0].set_title('%d models from %d chains' % (len(models), nchains))
        return fig




    def save_plots(self, refmodel=dict()):

        self.refmodel.update(refmodel)
        fig2a = self.plot_posterior_models1d(depint=0.01)
        self.plot_refmodel(fig2a, 'model', color='r', lw=1)
        self.savefig(fig2a, 'c_posterior_models1d_zoom.tif')
        fig2b = self.plot_posterior_models2d(depint=0.01)
        self.plot_refmodel(fig2b, 'model', color='red', lw=0.5, alpha=0.7)
        self.savefig(fig2b, 'c_posterior_models2d_zoom.tif')

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
dep, vs = np.loadtxt('observed/shallow4_mod.dat', usecols=[0, 2], skiprows=1).T
pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
pvs = np.repeat(vs, 2)

#truenoise = np.concatenate(([noise[0]], [np.std(ysw_err)],   # target 1
#                            [noise[2]], [np.std(yrf_err)],   # target 2
#                            [noise[3]], [np.std(yrwe_err)]))  # target 3

#explike = SynthObs.compute_explike(yobss=[ysw, yrf, yrwe], ymods=[_ysw, _yrf, _yrwe],
#                                   noise=truenoise, gauss=[False, True, False],
#                                   rcond=initparams['rcond'])
truemodel = {'model': (pdep, pvs),
             'nlays': 3,
#             'noise': truenoise,
#             'explike': explike,
             }

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
obj.save_plots(refmodel=truemodel)
