"""Plot of functional connectivity dynamics."""
import numpy as np
import pandas as pd

from brainets.io import load_marsatlas, set_log_level
from brainets.utils import normalize
from brainets.plot.circle import (plot_connectivity_circle_arrows,
                                  circular_layout)

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt



LOBE_COLOR = {'Subcortical': [1.0, 0.55, 0.46], 'Frontal': [0.0, 0.8, 0.64],
              'Parietal': [0.28, 0.5, 0.61], 'Occipital': [0.66, 0.53, 0.75],
              'Temporal': [1.0, 0.35, 0.42]}

LOBE_COLOR = {
    'Subcortical': [0.95, 0.25, 0.15],  # Stronger, deeper red  
    'Frontal': [0.0, 0.7, 0.5],         # Richer teal-green  
    'Parietal': [0.1, 0.4, 0.6],        # Deeper, more vivid blue  
    'Occipital': [0.55, 0.3, 0.75],     # Stronger purple with contrast  
    'Temporal': [0.9, 0.1, 0.2]         # Bolder, more saturated red-pink  
}

def _plot_fcd_stat(mi, pvalues, p, seeds=None):
    """Preprocess mi and p-values before plotting."""
    # get only where mi is significant
    if isinstance(pvalues, pd.DataFrame):
        pvalues.fillna(1., inplace=True)
    if pvalues is None:
        pvalues = pd.DataFrame(np.zeros(mi.shape), index=mi.index,
                               columns=mi.columns)
    if isinstance(seeds, list):
        all_s = np.array(mi.columns.tolist())[:, 0]
        is_seed = np.zeros((len(all_s), len(seeds)), dtype=bool)
        for n_s, s in enumerate(seeds):
            is_seed[:, n_s] = all_s == s
        is_seed = is_seed.any(axis=1)
        mi, pvalues = mi.loc[:, is_seed], pvalues.loc[:, is_seed]
    any_time = (np.array(pvalues) < p).any(0)
    mi = mi.loc[:, any_time]
    pvalues = pvalues.loc[:, any_time]

    return mi, pvalues


def plot_fcd_lineplot(mi, pvalues=None, p=.05, directed=True, vlines=None,
                      seeds=None):
    """Line plot of functional connectivity.

    Parameters
    ----------
    mi : pd.DataFrame
        Dataframe of the measure to plot (e.g mutual information). The
        dataframe should be organized as (n_times, n_roi)
    pvalues : pd.DataFrame | None
        Dataframe of p-values with the same shape as the measure
    p : float | 0.05
        Alpha value to use for only plotting significant results
    directed : bool | True
        Specify if the plot is directed or not
    vlines : dict | None
        Add and control vertical lines. If None, a single vertical lines is
        plotted at 0. For controlling vertical lines, use a dict like :

            * vlines={0.: dict(color='k'), 1.: dict(color='g', linestyle='--')}
              This draw two lines respectively at 0 and 1 secondes.

    Returns
    -------
    fig : plt.figure
        The matplotlib figure
    """
    # pre-process mi and p-values
    mi, pvalues = _plot_fcd_stat(mi, pvalues, p, seeds=seeds)

    # array conversion and extrema
    mi_arr, pv_arr = np.array(mi), np.array(pvalues)
    mi_min, mi_max = mi_arr.min(), mi_arr.max()
    is_signi = pv_arr < p

    # direction of y-labels
    direction = r'$\rightarrow$' if directed else r'$\leftrightarrow$'

    # build time indices and roi
    times = np.array(mi.index)
    roi = [f"{s}" + direction + f"{t}" for s, t in mi.columns]
    # if isinstance(seeds, list):
    #     roi = [f"{t}" for _, t in mi.columns]
    # else:
    #     roi = [f"{s}" + direction + f"{t}" for s, t in mi.columns]
    n_times, n_roi = len(times), len(roi)

    # plotting elements
    low_bound = np.zeros((n_times,))
    if not isinstance(vlines, dict):
        vlines = {0.: dict(color='k', linewidth=3)}

    fig, ax = plt.subplots(nrows=n_roi, figsize=(16, 12))
    ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax
    print(len(ax))
    print(ax[0])
    for n_r, r in enumerate(roi):
        # get the x variable and significant areas
        _x = np.array(mi.iloc[:, n_r])
        _p = is_signi[:, n_r]
        plt.sca(ax[n_r])
        plt.plot(times, _x, color='w', lw=3)
        plt.fill_between(times, low_bound, _x, color='#34495e')
        plt.fill_between(times[_p], low_bound[_p], _x[_p], color='#e74c3c')
        # plt.ylabel(r, rotation=0, fontweight='bold', va='top')
        ax[n_r].text(0, .2, r, fontweight="bold",
                ha="right", va="center", transform=ax[n_r].transAxes)
        # clean up axes
        ax[n_r].set_frame_on(False)
        is_last = n_r + 1 == n_roi
        if is_last:
            rm_axis = ['top', 'left', 'right']
            plt.xlabel('Time')
        else:
            rm_axis = ['top', 'left', 'right', 'left']
        for i in rm_axis:
            ax[n_r].spines[i].set_visible(False)
        plt.tick_params(axis='both', which='both', left=False, top=False,
                        labelleft=False, labelbottom=is_last, bottom=is_last)
        plt.ylim(mi_min, mi_max)
        plt.xlim(times[0], times[-1])
        if isinstance(vlines, dict):
            for n_v, kw_v in vlines.items():
                ax[n_r].axvline(n_v, **kw_v)
    fig.patch.set_visible(False)
    fig.subplots_adjust(hspace=-0.5, right=.99, bottom=.05, top=.99)

    return fig


def plot_fcd_image(mi, pvalues=None, p=0.05, directed=True, cmap='viridis',
                   vlines=None, fig=None, subplot=111, seeds=None, **kwargs):
    """Line plot of functional connectivity.

    Parameters
    ----------
    mi : pd.DataFrame
        Dataframe of the measure to plot (e.g mutual information). The
        dataframe should be organized as (n_times, n_roi)
    pvalues : pd.DataFrame | None
        Dataframe of p-values with the same shape as the measure
    p : float | 0.05
        Alpha value to use for only plotting significant results
    directed : bool | True
        Specify if the plot is directed or not
    vlines : dict | None
        Add and control vertical lines. If None, a single vertical lines is
        plotted at 0. For controlling vertical lines, use a dict like :

            * vlines={0.: dict(color='k'), 1.: dict(color='g', linestyle='--')}
              This draw two lines respectively at 0 and 1 secondes.

    Returns
    -------
    fig : plt.figure
        The matplotlib figure
    """
    # pre-process mi and p-values
    mi, pvalues = _plot_fcd_stat(mi, pvalues, p, seeds=seeds)

    # array conversion and extrema
    mi_arr, pv_arr = np.array(mi), np.array(pvalues)
    mi_min, mi_max = mi_arr.min(), mi_arr.max()
    is_signi = pv_arr < p
    mi_arr[~is_signi] = np.nan

    # direction of y-labels
    direction = r'$\rightarrow$' if directed else r'$\leftrightarrow$'

    # build time indices and roi
    times = np.array(mi.index)
    roi = [f"{s}" + direction + f"{t}" for s, t in mi.columns]
    n_times, n_roi = len(times), len(roi)
    roi_range = np.arange(len(roi) + 1)

    if not isinstance(vlines, dict):
        vlines = {0.: dict(color='k', linewidth=3)}

    # image plot
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    ax = plt.subplot(*subplot)
    plt.pcolormesh(times, roi_range, mi_arr.T, cmap=cmap, **kwargs)
    # ax = plt.gca()
    ax.set_yticks(np.arange(len(roi_range) - 1) + .5)
    ax.set_yticklabels(roi)
    plt.tight_layout()
    if isinstance(vlines, dict):
        for n_v, kw_v in vlines.items():
            ax.axvline(n_v, **kw_v)

    return fig


def plot_fcd_circular(mi, index, pvalues=None, p=0.05, subcortical=True,
                      cmap='YlOrRd', cmap_box='bwr', use_roi=None,
                      color_box_none='white', seeds=None, ma=None,
                      directed=True, **kwargs):
    """Circular plotting of functional connectivity using MarsAtlas.

    Parameters
    ----------
    mi : pd.DataFrame
        Dataframe of the measure to plot (e.g mutual information). The
        dataframe should be organized as (n_times, n_roi)
    index : float
        Time index to use for plotting
    pvalues : pd.DataFrame | None
        Dataframe of p-values with the same shape as the measure
    p : float | 0.05
        Alpha value to use for only plotting significant results
    subcortical : bool | True
        Plot subcortical roi or not
    cmap : string | 'YlOrRd'
        Colormap to use for arrows
    cmap_box : string | 'Spectral_r'
        Colormap to use for boxes. Boxes' colors refer to the net connectivity
        (i.e outcoming - incoming)
    color_box_none : string | 'white'
        Boxes color for empty boxes
    kwargs : dict | dict()
        Additional inputs are passed to the plot_connectivity_circle_arrows
        function

    Returns
    -------
    fig : plt.figure
        The matplotlib figure
    """
    from brainets.plot.circle import (plot_connectivity_circle_arrows,
                                      circular_layout)
    # -------------------------------------------------------------------------
    # load MarsAtlas informations
    # -------------------------------------------------------------------------
    if not isinstance(ma, pd.DataFrame):
        ma = load_marsatlas().rename(columns={'LR_Name': 'ROI'})
    if (use_roi is not None) and len(use_roi):
        is_roi = np.zeros((len(ma), len(use_roi)), dtype=bool)
        for n_r, r in enumerate(use_roi):
            is_roi[:, n_r] = ma['ROI'] == r
        ma = ma.loc[is_roi.any(1), :]
    ma_roi, ma_hemi = np.array(ma['ROI']), np.array(ma['Hemisphere'])
    ma_lobe = np.array(ma['Lobe'])
    if not subcortical:
        is_not_sub = ma_lobe != 'Subcortical'
        ma_roi, ma_hemi = ma_roi[is_not_sub], ma_hemi[is_not_sub]

    # -------------------------------------------------------------------------
    # remove non-significant links
    # -------------------------------------------------------------------------
    times = np.array(mi.index)
    if not isinstance(pvalues, pd.DataFrame):
        pvalues = pd.DataFrame(np.zeros(np.array(mi).shape), index=mi.index,
                               columns=mi.columns)
    # get mi at the selected time index
    pvalues.fillna(1., inplace=True)
    index_i = np.abs(times - index).argmin()
    mi_t = mi.iloc[index_i, :].reset_index()
    pv_t = pvalues.iloc[index_i, :].reset_index()
    mi_t = mi_t.pivot(index='source', columns='target', values=times[index_i])
    pv_t = pv_t.pivot(index='source', columns='target', values=times[index_i])
    # seed based plot
    if isinstance(seeds, list):
        sources_names = np.array(mi_t.index)
        is_source = np.ones((len(sources_names), len(seeds)), dtype=bool)
        for n_s, s in enumerate(seeds):
            is_source[:, n_s] = sources_names == s
        pv_t.iloc[~is_source.any(axis=1), :] = 1.
    # reorder sources / targets according to the default MarsAtlas order
    mi_t = mi_t.reindex(index=ma_roi, columns=ma_roi)
    pv_t = pv_t.reindex(index=ma_roi, columns=ma_roi)
    pv_t.fillna(1., inplace=True)
    mi_arr, pv_arr = np.array(mi_t), np.array(pv_t)

    # -------------------------------------------------------------------------
    # ravel the connectivity array as a 1D vector
    # -------------------------------------------------------------------------
    is_signi = pv_arr < p
    ind = np.where(is_signi)
    con = mi_arr[is_signi]

    # -------------------------------------------------------------------------
    # define roi order
    # -------------------------------------------------------------------------
    # Get the unique list of ROIs (i.e only those in the left hemisphere) :
    is_left = ma_hemi == 'L'
    u_roi = ma_roi[is_left]
    # Get the ROI order to use in the plot :
    roi_l = [f'L_{k[2::]}' for k in u_roi]
    roi_r = [f'R_{k[2::]}' for k in u_roi]
    roi_order = roi_l[::-1] + roi_r
    roi_label = [k[2::] for k in ma_roi]

    # -------------------------------------------------------------------------
    # box coloring
    # -------------------------------------------------------------------------
    # start by hiding everywhere the array is not significant and take the net
    # (outcoming - incoming) of remaining arrows.
    mi_masked = np.ma.masked_array(mi_arr, mask=~is_signi)
    incoming, outcoming = np.nansum(mi_masked, 0), np.nansum(mi_masked, 1)
    con_net = (outcoming.data - incoming.data)
    # now, mask everywhere there's no arrow
    con_mask = np.c_[is_signi.any(0), is_signi.any(1)].any(1)
    col_values = np.ma.masked_array(con_net, mask=~con_mask)
    # col_values = normalize(col_values, -1, 1)
    cmap_box  = cm.get_cmap(cmap_box)
    cmap_box.set_bad(color_box_none)
    colors = cmap_box(col_values + .5).tolist()

    # -------------------------------------------------------------------------
    # compute circular plot
    # -------------------------------------------------------------------------
    # get the nodes angles
    node_angles = circular_layout(ma_roi, roi_order, start_pos=90,
                                  group_boundaries=[0, len(ma_roi) / 2])
    # automatic colorbar limits
    kwargs['vmin'] = kwargs.get('vmin', con.min())
    kwargs['vmax'] = kwargs.get('vmax', con.max())
    # circular plot
    fig, _ = plot_connectivity_circle_arrows(
        con, roi_label, indices=ind, node_angles=node_angles,
        node_colors=colors, colormap=cmap, arrow=directed, **kwargs)

    return fig


def plot_fcd_circular_xr(da, source_name='sources', target_name='targets',
                         directed=False, cmap='YlOrRd', node_color='lobe',
                         ma=None, cmap_node='viridis', node_none='whitesmoke',
                         lobes=None, **kwargs):
    """Circular plotting of functional connectivity using MarsAtlas and xarray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray of shape (n_roi, n_roi)
    source_name : string | 'sources'
        Coordinate name to consider for sources
    target_name : string | 'targets'
        Coordinate name to consider for targets
    directed : bool | False
        Directed connections or not. If False, the upper triangle is going to
        be used
    cmap : string | 'YlOrRd'
        Colormap to use for arrows
    node_color : string | 'lobe'
        Option to color the box in front of each roi. Choose either :

            * 'lobe' : one color per lobe
            * 'density' : color according to the number of connections with
              this node (hub)
    ma : pd.DataFrame | None
        Custom reference dataframe
    cmap_node : string | 'Spectral_r'
        Colormap to use for boxes. Only valid when using 'density' option
    node_none : string | 'whitesmoke'
        Boxes color for empty boxes
    lobes : list | None
        Lobe to use ('Frontal', 'Parietal', 'Occipital', 'Temporal',
        'Succortical')
    kwargs : dict | dict()
        Additional inputs are passed to the plot_connectivity_circle_arrows
        function

    Returns
    -------
    fig : plt.figure
        The matplotlib figure
    """
    import xarray as xr
    assert isinstance(da, xr.DataArray) and (da.ndim == 2)
    assert da.shape[0] == da.shape[1]
    assert node_color in ['lobe', 'density', 'weight']

    # be sure of dimension names
    # da = da.rename({source_name: 'sources', target_name: 'targets'})
    da = da.set_index({source_name: 'sources', target_name: 'targets'})
    # compute net weights from c_in and c_out
    roi = da.sources.data
    c_in = da.sum(dim='sources')
    c_out = da.sum(dim='targets')
    if directed:
        w = c_out.data - c_in.data
    else:
        w = c_out.data
    # set to nan lower triangle if not directed
    if not directed:
        idx_tri = np.triu_indices_from(da.data)
        da.data[idx_tri[0], idx_tri[1]] = np.nan
    # stack links
    da = da.stack(roi=('sources', 'targets')).dropna('roi')
    sources, targets = da['sources'].data, da['targets'].data

    # load marsatlas table and select lobes
    if ma is None:
        ma = load_marsatlas()
    ma_backup = ma.copy()
    if isinstance(lobes, str): lobes = [lobes]
    if isinstance(lobes, list):
        is_lobe = ma['Lobe'].str.contains('|'.join(lobes))
        ma = ma.loc[is_lobe]

    # build a multiindex
    ma.index = pd.MultiIndex.from_arrays((ma['Hemisphere'], ma['Lobe']),
                                         names=('Hemisphere', 'Lobe'))
    ma_roi = list(ma.loc['L']['LR_Name'])[::-1] + list(ma.loc['R']['LR_Name'])
    roi_order = ma_roi
    node_names = [k[2::] for k in ma_roi]

    # find indices of sources and targets
    s_idx, t_idx = [], []
    for s, t in zip(sources, targets):
        s_idx += [ma_roi.index(s)]
        t_idx += [ma_roi.index(t)]
    ind = (np.array(s_idx), np.array(t_idx))


    node_angles = circular_layout(ma_roi, roi_order, start_pos=90,
                                  group_boundaries=[0, len(ma_roi) / 2])
    node_colors = ['k'] * len(node_names)

    if node_color == 'lobe':
        ma_lobes = ma_backup.set_index('LR_Name').loc[ma_roi]['Lobe']
        node_colors = [LOBE_COLOR[k] for k in ma_lobes]
    elif node_color == 'density':
        # get the normalized count of each roi
        unique, counts = np.unique(np.r_[sources, targets], return_counts=True)
        counts_n = (counts - counts.min())
        counts_n = counts_n / counts_n.max()
        # value to color
        map_color = plt.get_cmap(cmap_node)
        color = {k: map_color(v) for k, v in zip(unique, counts_n)}
        node_colors = [color[k] if k in unique else node_none for k in ma_roi]
    elif node_color == 'weight':
        # get the normalized weight for each roi
        w = np.nan_to_num(w)
        if directed:
            w = (w/np.abs(w).max()+1)/2
        else:
            w = w/np.abs(w).max()
        # value to color
        map_color = plt.get_cmap(cmap_node)
        color = {k: map_color(v) for k, v in zip(roi, w)}
        node_colors = [color[k] for k in ma_roi]

    # create figure and ax
    fig, ax = plot_connectivity_circle_arrows(
        da.data, node_names, indices=ind, node_angles=node_angles,
        colormap=cmap, arrow=directed, node_colors=node_colors, **kwargs)
    
    return fig, ax


def plot_heatmap_lines(df, columns, ax=None, **kw):
    """Draw horizontal and vertical lines arround anatomical delimitations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the anatomical informations
    columns : string, list
        The columns to use as the delimiters (e.g 'Lobe', ['Lobe',
        'Hemisphere'])
    ax : axes | None
        Axes to use the lines
    kw : dict | {}
        Additional inputs are sent to the plt.plot function
    """
    # extract columns of the dataframe
    if isinstance(columns, str):
        columns = [columns]
    classifier = list(df[columns].sum(axis=1))
    # string to integer conversion
    cla_u = np.unique(classifier)
    cla_i = pd.Series(classifier).replace({s: i for i, s in enumerate(cla_u)},
                                          regex=True)
    split = (np.where(np.diff(cla_i) != 0)[0] + 1).tolist()
    split = [0] + split + [len(classifier)]
    pairs = np.c_[split[:-1], split[1:]]

    if ax is None:
        ax = plt.gca()

    for k in range(len(split) - 2):
        ax.plot((pairs[k], pairs[k]), (pairs[k], pairs[k + 1]), **kw)
        ax.plot((pairs[k + 1], pairs[k + 1]), (pairs[k], pairs[k + 1]), **kw)
        ax.plot((pairs[k], pairs[k + 1]), (pairs[k], pairs[k]), **kw)
        ax.plot((pairs[k], pairs[k + 1]), (pairs[k + 1], pairs[k + 1]), **kw)
