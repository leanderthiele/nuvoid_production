from sys import argv
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

import corner

from _plot_get_chain import get_chain
from _plot_labels import plot_labels
from _plot_style import *
from _plot_fiducials import fid

Nbins = 50

class Formatter :
    
    def __init__ (self,
                  have_hash=False, have_stats=True, have_kmax=False, have_budget=True,
                  have_vsf_info=False, have_vgplk_info=False,
                  fs_color=black, fid_color='grey', special=None, one_fid_label=True,
                  default_colors=True) :
        self.have_hash = have_hash
        self.have_stats = have_stats
        self.have_kmax = have_kmax
        self.have_budget = have_budget
        self.have_vsf_info = have_vsf_info
        self.have_vgplk_info = have_vgplk_info
        self.fs_color = fs_color
        self.fid_color = fid_color
        self.special = special
        self.one_fid_label = one_fid_label
        self.default_colors = default_colors
        self.reset()

    def reset(self) :
        # reset for new axis
        self.used_fid_label = False
        self.color_cycle = cycle(default_colors)
        self.fs_line_cycle = cycle(default_linestyles)

    def __call__ (self, chain_container) :

        plot_kwargs = {}
        info = []

        if chain_container.is_fs :
            info.append('EFTofLSS')
            if self.fs_color is not None :
                plot_kwargs['color'] = self.fs_color
                plot_kwargs['linestyle'] = next(self.fs_line_cycle)
        if self.have_hash and chain_container.quick_hash is not None :
            info.append(f'$\\tt{{ {chain_container.quick_hash} }}$')
        if self.have_stats :
            info.append(f'{chain_container.stats_str}')
        if self.have_kmax and chain_container.kmax is not None :
            info.append(f'$k_{{\sf max}}={chain_container.kmax:.2f}$')
        if not chain_container.is_fs :
            if self.have_budget and 'sim_budget' in chain_container.model_settings :
                info.append(f'budget={chain_container.model_settings["sim_budget"]*100:.0f}%')
            if chain_container.fid_idx is not None :
                info.append(f'fiducials')
                plot_kwargs['color'] = self.fid_color
            if chain_container.compression_settings['use_vsf'] and self.have_vsf_info :
                if chain_container.compression_settings['vsf_Rmin']==30 :
                    if chain_container.compression_settings['vsf_Rmax']==80 :
                        info.append('all $R_{\sf void}$')
                    else :
                        info.append(f'$R_{{\sf void}} < {chain_container.compression_settings["vsf_Rmax"]}$')
                else :
                    assert chain_container.compression_settings['vsf_Rmax']==80, 'not implemented'
                    info.append(f'$R_{{\sf void}} > {chain_container.compression_settings["vsf_Rmin"]}$')
                if 0 in chain_container.compression_settings['vsf_zbins'] :
                    if 1 in chain_container.compression_settings['vsf_zbins'] :
                        pass
                        # info.append('all $z$')
                    else :
                        info.append('$z < 0.53$')
                else :
                    assert 1 in chain_container.compression_settings['vsf_zbins']
                    info.append('$z > 0.53$')
            if chain_container.compression_settings['use_vgplk'] and self.have_vgplk_info :
                if len(chain_container.compression_settings['vgplk_Rbins']) ==  3 :
                    info.append('all $R_{\sf min}$')
                else :
                    info.append(f'$R_{{\sf min}} = {{ {",".join(map(str, sorted(chain_container.compression_settings["vgplk_Rbins"])))} }}$')

        if chain_container.fid_idx is None or not self.used_fid_label :
            plot_kwargs['label'] = ', '.join(info)

        if chain_container.fid_idx is not None and self.one_fid_label :
            self.used_fid_label = True

        if self.special is not None :
            d = self.special(chain_container)
            for k, v in d.items() :
                plot_kwargs[k] = v

        if 'color' not in plot_kwargs or plot_kwargs['color'] is None :
            if self.default_colors and chain_container.compression_settings is not None :
                cidx = 3 if all(chain_container.compression_settings[f'use_{s}'] \
                                for s in ['vsf', 'vgplk', 'plk',]) \
                       else 2 if all(chain_container.compression_settings[f'use_{s}'] \
                                     for s in ['vgplk', 'plk', ]) \
                       else 1 if all(chain_container.compression_settings[f'use_{s}'] \
                                     for s in ['vsf', 'plk', ]) \
                       else 0
                plot_kwargs['color'] = default_colors[cidx]
            else :
                plot_kwargs['color'] = next(self.color_cycle)

        return plot_kwargs


def plot_cdf (runs, ax, formatter=Formatter(), param_name='Mnu', pretty=True, want_corner=False,
              levels_above=True) :

    formatter.reset()
    
    chain_containers = [get_chain(run) for run in runs]
    xmin = min(c.priors[param_name][0] for c in chain_containers)
    xmax = max(c.priors[param_name][1] for c in chain_containers)
    edges = np.linspace(xmin, xmax, num=Nbins+1)

    fig_corner = None
    ax_corner_legend = None

    if want_corner :
        assert all(chain_containers[0].param_names == c.param_names for c in chain_containers)
        rng = np.random.default_rng(137) # for downsampling to same length
        min_len = min(len(c.chain) for c in chain_containers)

    for chain_container in chain_containers :
        x = chain_container.chain[:, chain_container.param_names.index(param_name)]

        level = 0.95
        cl = np.quantile(x, 0.95)
        print(f'{chain_container.quick_hash}: {level} quantile = {cl}')

        cdf = np.array([np.count_nonzero(x<e) for e in edges]) / len(x)
        plot_kwargs = formatter(chain_container)
        ax.plot(edges, cdf, **plot_kwargs)

        if want_corner :
            # corner doesn't normalize by itself so we need to downsample all chains to the same length
            corner_chain = chain_container.chain
            if len(corner_chain) != min_len :
                corner_chain = rng.choice(corner_chain, size=min_len, replace=False)
            fig_corner = corner.corner(corner_chain, 
                                       labels=[plot_labels[p] for p in chain_container.param_names],
                                       plot_datapoints=False, plot_density=False, no_fill_contours=True,
                                       levels=1 - np.exp(-0.5 * np.array([2])**2), # values in array are sigmas
                                       color=plot_kwargs['color'],
                                       hist_bin_factor=1,
                                       fig=fig_corner)
            if ax_corner_legend is None :
                nparams = corner_chain.shape[-1]
                ax_corner_legend = fig_corner.axes[1 if nparams<=3 else 2]
            if 'label' in plot_kwargs :
                ax_corner_legend.plot(np.random.rand(2), label=plot_kwargs['label'], color=plot_kwargs['color'])


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)

    if pretty :
        ax.set_xlabel(plot_labels[param_name])
        ax.set_ylabel('$P(<)$')
        ax.legend(loc='lower right', frameon=False)
        ax.axline((xmin, 0), (xmax, 1), color='grey', linestyle='dashed')
        if any(c.fid_idx is not None for c in chain_containers) :
            ax.axvline(fid[param_name], color='grey', linestyle='dotted')
            ax.text(fid[param_name], 0, ' fiducial', ha='right', va='bottom', rotation=90)
        for percentile in [68, 95, ] :
            y = percentile / 100
            ax.axhline(y, color='grey', linestyle='dotted')
            ax.text(xmin, y+(0 if levels_above else -0.02), f' {percentile}%',
                    ha='left', va='bottom' if levels_above else 'top', transform=ax.transData)

    if want_corner :
        ax_corner_legend.set_xlim(10,11)
        ax_corner_legend.legend(loc='upper left', frameon=False)

    return fig_corner



def plot_cdfs (runs, name) :
    # runs is a list of tuples, where [0] = list of runs,
    # and [optional 1] = dict(formatter, param_name, title)
    # or it is just a tuple

    if isinstance(runs, tuple) :
        runs = [runs, ]
    if '_' in name :
        name, width, height = name.split('_')
        width = float(width)
        height = float(height)
    else :
        width, height = 5*np.sqrt(len(runs)), 5/np.sqrt(len(runs))
    levels_above = height > 3.9
    fig, ax = plt.subplots(nrows=1, ncols=len(runs), figsize=(width, height))
    try :
        ax = ax.flatten()
    except AttributeError :
        ax = [ax, ]

    for ii, (r, a) in enumerate(zip(runs, ax)) :
        kw = {}
        title = None
        if len(r) > 1 :
            assert len(r) == 2
            for s in ['formatter', 'param_name', 'want_corner', ] :
                if s in r[1] :
                    kw[s] = r[1][s]
            if 'title' in r[1] :
                title = r[1]['title']
        fig_corner = plot_cdf(r[0], a, levels_above=levels_above, **kw)
        if ii != 0 :
            a.set_ylabel(None)
        if title is not None :
            a.set_title(title)

    savefig(fig, f'cdf_{name}')
    if fig_corner is not None :
        fig_corner.set_figheight(7)
        fig_corner.set_figwidth(7)
        N_ = np.sqrt(len(fig_corner.axes))
        delta_ = 0.7 * (N_ / 9)
        for a in fig_corner.axes :
            a.xaxis.set_label_coords(0.5, -delta_)
            a.yaxis.set_label_coords(-delta_, 0.5)
        savefig(fig_corner, f'posterior_{name}')


if __name__ == '__main__' :
    
    todo = {
            'statsvoids':
            ([
              'lfi_chain_v0_dd916201431a1b9e5b960c075709f418_6b656a4fa186194104da7c4f88f1d4c2_emcee.npz',
              'lfi_chain_v0_c62bf69edab920b916fbca0a9cd81acd_6b656a4fa186194104da7c4f88f1d4c2_emcee.npz',
              'lfi_chain_v0_37715c02cbc4c059eaac51410906acd8_6b656a4fa186194104da7c4f88f1d4c2_emcee.npz',
             ],
            ),
            'statswPgg_5_4':
            ([
              'lfi_chain_v0_a8e282250ab78bf4fac45f297b4d822c_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_6604ce64512d9fb9575ec29edad6d652_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_fe169f0711650a7bc28866af6738b56d_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue',
              # these are just for testing the effect of EFT prior
              # 'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue_prior2',
              # 'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue_prior4',
             ],
            ),
            'largevoids_5_2':
            ([
              'lfi_chain_v0_a8e282250ab78bf4fac45f297b4d822c_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_2ba65ec0068b7db92a57d6b7df935d17_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_7ef5da19b69c33bd891d7c6ca67453cb_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
             ],
             {'formatter': Formatter(have_vsf_info=True,
                                     special=lambda c: {'linestyle': '-' if not c.compression_settings['use_vsf']
                                                                            or c.compression_settings['vsf_Rmin']==30
                                                                     else '-.' if c.compression_settings['vsf_Rmin']==40
                                                                     else ':' if c.compression_settings['vsf_Rmin']==50
                                                                     else '--' }), }
            ),
            'cmpEFT':
            ([
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'full_shape_production_kmin0.01_kmax0.15_lmax4',
              'full_shape_production_kmin0.01_kmax0.2_lmax4',
             ],
             {'formatter': Formatter(have_kmax=True), }
            ),
            'quadrupole_5_2':
            ([
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_6aad59fa700e94d8cedc0ec994380573_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue',
              'full_shape_production_kmin0.01_kmax0.15_lmax2_APTrue',
             ],
             {'formatter': Formatter(special=lambda c: {'linestyle': '-' if c.lmax==0 else '--', }), }
            ),
            'budget_5_2':
            ([
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_ce02407fa6db4df6343a60fe19a6f4c7_emcee_wweight.npz',
             ],
             {'formatter': Formatter(default_colors=False), }
            ),
            'kmax_5_4':
            ([
              'lfi_chain_v0_a8e282250ab78bf4fac45f297b4d822c_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue',
              'full_shape_production_kmin0.01_kmax0.2_lmax0_APTrue',
             ],
             {'formatter': Formatter(have_kmax=True,
                                     special=lambda c: {'linestyle': '-' if c.kmax<0.17 else '--', }), }
            ),
            'fid':
            [
            ([
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              *[f'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_fid{ii}_emceegpu.npz'
                for ii in [1208, 1260, 1295, 1944, 2261, 2302, 2378, 3389, 346,
                           4086, 4819, 4844, 5168, 5259, 5832, 808, 97, ]
               ],
             ],
             {'title': '$k_{\sf max}=0.15$',
              'formatter': Formatter(have_stats=False,
                                     special=lambda c: {'label': 'data' if c.fid_idx is None
                                                            else 'fiducials' if c.fid_idx==1208
                                                            else None,
                                                        'marker': 'o' if c.fid_idx is None else None})
             }
            ),
            ([
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              *[f'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_fid{ii}_emceegpu.npz'
                for ii in [1228, 1581, 1837, 1882, 1911, 1998, 2530, 2658, 2808, 3066,
                           3195, 3329, 3397, 433, 4640, 5350, 5458, 6189, 6579, ]
               ],
             ],
             {
              'title': '$k_{\sf max}=0.20$',
              'formatter': Formatter(have_stats=False,
                                     special=lambda c: {'label': 'data' if c.fid_idx is None
                                                            else 'fiducials' if c.fid_idx==1228
                                                            else None,
                                                        'marker': 'o' if c.fid_idx is None else None})
             }
            ),
            ],
            'fid1':
            ([
              'lfi_chain_v0_a8e282250ab78bf4fac45f297b4d822c_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz',
              'lfi_chain_v0_6604ce64512d9fb9575ec29edad6d652_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz',
              # NOTE that faae run on different seeds but it should be ok
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz',
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_fid*_emceegpu.npz',
             ],
             {'formatter': Formatter(fid_color=None, one_fid_label=False, have_kmax=True,
                                     special=lambda c: {'linestyle': '-' if c.kmax<0.17 else '--', }, ), }
            ),
            'voidcuts':
            [
            ([
              'lfi_chain_v0_dd916201431a1b9e5b960c075709f418_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_143746cce89185cf116f87d452bb85a0_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_e5debc1a5af3f0cad12b97cd8cdc96a2_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_183edfcf2596ab34bc6853a235a4312f_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_4e6cfeceee9d3a96b66af8f3920979b1_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
             ],
             {'title': '$N_v$ cutting', 'formatter': Formatter(have_vsf_info=True, default_colors=False), }
            ),
            (
             [
              'lfi_chain_v0_c62bf69edab920b916fbca0a9cd81acd_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_7e2cfa24acd42dc91e781f0352bcda76_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_c86572390de35979ffe32343bcae263b_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_505845a955640811c282b6e9d6912957_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
             ],
             {'title': '$P^{vg}$ cutting', 'formatter': Formatter(have_vgplk_info=True, default_colors=False), }
            ),
            ],
            'hod':
            ([
              'lfi_chain_v0_a8e282250ab78bf4fac45f297b4d822c_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_6604ce64512d9fb9575ec29edad6d652_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_0b59eb1479fd93eaeb7262ce1a805d63_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_faae54307696ccaff07aef77d20e1c1f_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
             ],
             {'want_corner': True, 'formatter': Formatter(have_kmax=True, default_colors=False), }
            ),
            'moreparams':
            ([
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_4c949c410d4ac209fb515137929b7c7b_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_4c949c410d4ac209fb515137929b7c7b_emcee_wweight.npz',
             ],
             {'want_corner': True, 'formatter': Formatter(have_kmax=True, default_colors=False), }
            ),
            'moreparams1':
            ([
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_ddd913a3f968fe60c439465ee476c3c9_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_ddd913a3f968fe60c439465ee476c3c9_emcee_wweight.npz',
             ],
             {'want_corner': True, 'formatter': Formatter(have_kmax=True, default_colors=False), }
            ),
            'eftbias':
            ([
              'full_shape_production_kmin0.01_kmax0.15_lmax0_APTrue',
              'full_shape_production_kmin0.01_kmax0.15_lmax4',
              'full_shape_production_kmin0.01_kmax0.2_lmax0_APTrue',
              'full_shape_production_kmin0.01_kmax0.2_lmax4',
             ],
             {'want_corner': True, 'formatter': Formatter(have_kmax=True, fs_color=None), }
            ),
            'differentparams':
            ([
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_4c949c410d4ac209fb515137929b7c7b_emcee_wweight.npz',
              'lfi_chain_v0_8c442ad9200d17242e8e97227366fac9_ddd913a3f968fe60c439465ee476c3c9_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_6b656a4fa186194104da7c4f88f1d4c2_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_4c949c410d4ac209fb515137929b7c7b_emcee_wweight.npz',
              'lfi_chain_v0_deee27266999e84b46162bf7627d71b6_ddd913a3f968fe60c439465ee476c3c9_emcee_wweight.npz',
             ],
             {'formatter': Formatter(default_colors=False, have_kmax=True, special=lambda c: {'linestyle': '-' if c.kmax<0.17 else '--', },), }
            ),
            'voidsonly':
            ([
              'lfi_chain_v0_cba1046bfb1c70008ec7fe0d86945ed2_6b656a4fa186194104da7c4f88f1d4c2_emcee.npz',
              'lfi_chain_v0_c62bf69edab920b916fbca0a9cd81acd_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_7e2cfa24acd42dc91e781f0352bcda76_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_c86572390de35979ffe32343bcae263b_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
              'lfi_chain_v0_505845a955640811c282b6e9d6912957_6f103cb42a1d934d3314f5429fb7aa9a_emcee.npz',
             ],
             {'formatter': Formatter(have_vgplk_info=True, default_colors=False), }
            ),
           }
    
    try :
        only_this = argv[1]
    except IndexError :
        only_this = None

    for name, runs in todo.items() :
        if only_this is None or name.split('_')[0] == only_this :
            print(name)
            plot_cdfs(runs, name)
