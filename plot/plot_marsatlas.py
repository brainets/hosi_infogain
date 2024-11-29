"""2d plot of the gcmi."""
import logging
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from brainets.io import load_marsatlas, set_log_level, is_xarray_installed

logger = logging.getLogger('brainets')

mpl.rcParams['grid.linestyle'] = "--"
mpl.rcParams['grid.color'] = "k"


def plot_marsatlas(df, pvalues=None, threshold=.05, time=None,
                   subcortical=False, contrast=5, cmap='hot_r', title=None,
                   cblabel=None, vlines=None, grid=False, merge_lr=False,
                   ma=None, verbose=None):
    """Plot data sorted using MarsAtlas parcellation.

    This function sort the data by hemisphere, lobe (frontal, occipital,
    parietal, temporal and subcortical) and by roi [1]_.

    Parameters
    ----------
    df : DataFrame
        The data to plot using MarsAtlas labels. `df` must be a pandas
        DataFrame of shape (n_pts, n_roi). The DataFrame indices (df.index) is
        then going to be considered as the time vector and columns (df.columns)
        as the ROI to plot.
    pvalues : DataFrame | None
        P-values associated to the data. `pvalues` must also be a DataFame with
        the same number of time indices and columns as the `df` input
        parameter.
    threshold : float | .05
        Threshold for selecting p-values. Every data points with a p-value
        exceding `threshold` is going to be ignored in the plot
    time : list | tuple | None
        Time boundaries to use. Should be (time_start, time_end).
    subcortical : bool | False
        Add subcortical region of interest (True) or not (False)
    contrast : int | float
        Contrast to use for the plot. A contrast of 5 means that vmin is set to
        5% of the data and vmax 95% of the data. If None, vmin and vmax are set
        to the min and max of the data. Alternatively, you can also provide
        a tuple to manually define it
    title : string | None
        Title of the figure
    cblabel : string | None
        Colorbar label
    vlines : dict | None
        Add and control vertical lines. If None, two vertical lines are plotted
        at -0.25 and 0. For controlling vertical lines, use a dict like :

            * vlines={0.: dict(color='k'), 1.: dict(color='g', linestyle='--')}
              This draw two lines respectively at 0 and 1 secondes.
    grid : bool | False
        Add a grid. If then you want to control your grid look, you can
        `import matplotlib as mpl` and modify grid settings :

            * mpl.rcParams['grid.linestyle'] = "--"
            * mpl.rcParams['grid.color'] = "w"
    cmap : string | 'viridis'
        The colormap to use
    merge_lr : bool | False
        Merge the left and right hemispheres into a single one. In order to
        work, the ROI should be formatted from 'L_OFCv' to 'OFCv' and they
        should all be uniques
    ma : pd.DataFrame | None
        A dataframe table for replacing the default MarsAtlas table. The `ma`
        input should be a DataFrame with the columns `ROI`, 'Lobe' and
        'Hemisphere'

    Returns
    -------
    fig : plt.figure
        Matplotlib figure with MarsAtlas parcels

    References
    ----------
    .. [1] Auzias, G., Coulon, O., & Brovelli, A. (2016). MarsAtlas: a cortical
       parcellation atlas for functional mapping. Human brain mapping, 37(4),
       1573-1592.
    """
    set_log_level(verbose)

    # checkout that inputs are DataArray
    if is_xarray_installed():
        from xarray import DataArray
        if isinstance(df, DataArray):
            df = da_to_df(df)
        if isinstance(pvalues, DataArray):
            pvalues = da_to_df(pvalues)
    assert isinstance(df, pd.DataFrame) and (df.ndim == 2)
    df.rename(columns={'LR_Name': 'ROI'}, inplace=True)

    # Load MarsAtlas DataFrame
    logger.info('    Load MarsAtlas labels')
    if not isinstance(ma, pd.DataFrame):
        df_ma = load_marsatlas().rename(columns={'LR_Name': 'ROI'})
    else:
        df_ma = ma
    if merge_lr:
        # select only the left hemisphere
        df_ma = df_ma.loc[df_ma['Hemisphere'] == 'L'].reset_index(drop=True)
        df_ma.replace({'L\\_': '', 'R\\_': ''}, regex=True, inplace=True)
        n_c, n_r, figsize = 30, 11, (12, 10)
    else:
        n_c, n_r, figsize = 30, 23, (16, 10)
    df = df.reindex(columns=df_ma['ROI'], fill_value=np.nan)

    # pvalues
    if isinstance(pvalues, pd.DataFrame):
        logger.info("    Consider p-values < %.2f" % threshold)
        pvalues.rename(columns={'LR_Name': 'ROI'}, inplace=True)
        pvalues = pvalues.reindex(columns=df_ma['ROI'], fill_value=np.nan)
        pvalues.fillna(1., inplace=True)
        if not np.any(np.array(pvalues) < threshold):
            logger.error("Nothing came-out significant. Sorry :(")
            return None
        df[pvalues >= threshold] = np.nan

    # Built the multi-indexing
    assert len(df.columns) == len(df_ma)
    mi = pd.MultiIndex.from_frame(df_ma[['Hemisphere', 'Lobe', 'ROI']])
    df.columns = mi

    # Scale colormap to cortical values if subcortical is False
    if subcortical == False:
        df.loc[:, ('L', 'Subcortical')] = np.nan
        df.loc[:, ('R', 'Subcortical')] = np.nan

    # Time vector
    if isinstance(time, (list, tuple, np.ndarray)) and (len(time) == 2):
        logger.info("    Time selection between "
                    "(%.2f, %.2f)" % (time[0], time[1]))
        df = df.loc[time[0]:time[1], :]

    # Get colorbar limits
    if isinstance(contrast, (int, float)):
        nnz = np.isfinite(df)
        vmin = np.percentile(np.array(df)[nnz], contrast)
        vmax = np.percentile(np.array(df)[nnz], 100 - contrast)
    elif isinstance(contrast, (tuple, list)) and (len(contrast) == 2):
        vmin, vmax = contrast
    else:
        vmin, vmax = np.nanmin(df), np.nanmax(df).max()
    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    # Generate plots
    if vlines is None:
        vlines = {0.: dict(color='w', linewidth=3),
                  -0.25: dict(linestyle='--', color='w', linewidth=2)
                  }
    title = '' if not isinstance(title, str) else title

    # plot with / without subcortical regions
    return _plot_ma(df, title, figsize, (n_r, n_c), vlines, cblabel, grid,
                    subcortical, merge_lr, **kwargs)


###############################################################################
###############################################################################
#                            PLOTTING FUNCTIONS
###############################################################################
###############################################################################

def _plot_ma(df, title, figsize, figdim, vlines, cblabel,
             grid, subcortical, merge_lr, **kwargs):
    """Plot for a single hemisphere without subcortical regions."""
    fig = plt.figure(figsize=figsize)
    n_r, n_c = figdim
    if isinstance(title, str):
        fig.suptitle(title, fontweight='bold', fontsize=15, y=.99)
    gs = GridSpec(n_c, n_r, left=0.05, bottom=0.03, right=0.99, top=0.90,
                  wspace=.01)
    time = df.index
    # define subplots
    if merge_lr:
        subplots = {
            'Temporal': {'L' : gs[21:26, 0:5]},
            'Frontal': {'L' : gs[0:19, 0:5]},
            'Parietal': {'L' : gs[14:23, 6:11]},
            'Occipital': {'L' : gs[25:29, 6:11]},
            'Subcortical': {'L' : gs[6:12, 6:11]}}
        cb_gs = gs[-2, 0:5]
        hemi = ['L']
    else:
        subplots = {
            'Temporal': {'L' : gs[21:26, 0:5], 'R' : gs[21:26, 18:23]},
            'Frontal': {'L' : gs[0:19, 0:5], 'R' : gs[0:19, 18:23]},
            'Parietal': {'L' : gs[13:22, 6:11], 'R' : gs[13:22, 12:17]},
            'Occipital': {'L' : gs[25:29, 6:11], 'R' : gs[25:29, 12:17]},
            'Subcortical': {'L' : gs[6:12, 6:11], 'R' : gs[6:12, 12:17]}}
        cb_gs = gs[-2, 19:22]
        hemi = ['L', 'R']
    if not subcortical: subplots.pop('Subcortical')  # noqa

    for l in subplots.keys():
        for h in hemi:
            ax = plt.subplot(subplots[l][h])
            _plot_single_subplot(ax, df, time, h, l, vlines=vlines,
                                 xticks=True, grid=grid, **kwargs)


    # -------------------------------------------------------------------------
    # Colorbar
    ax11 = plt.subplot(cb_gs)
    norm = mpl.colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    cb_cmap = mpl.cm.get_cmap(kwargs['cmap'])
    mpl.colorbar.ColorbarBase(ax11, cmap=cb_cmap, norm=norm, label=cblabel,
                              orientation='horizontal')

    # This is a bug of mpl, but plt.tight_layout doesn't work...
    fig.set_tight_layout(True)
    return fig


def _plot_single_subplot(ax, df, time, hemi, lobe, xticks=False, vlines=None,
                         title=None, grid=False, **kwargs):
    """Generate the plot of a single subplot."""
    levels = df.keys().levels
    if (hemi not in levels[0]) or (lobe not in levels[1]):
        logger.warning("Nothing in the %s %s lobe" % (hemi, lobe))
        ax.axis('off')
        return None
    # Get the data for this hemisphere / lobe
    _df = df[hemi][lobe]
    data, yticks = np.array(_df), list(_df.keys())
    # Build the colormap for bad values
    data = np.ma.masked_array(data, mask=~np.isfinite(data))
    cmap = copy.copy(mpl.cm.get_cmap(kwargs['cmap']))
    cmap.set_bad(color='white')
    kwargs['cmap'] = cmap
    kwargs['shading'] = 'auto'
    # Make the plot
    n_t = len(yticks)
    yvec = np.arange(n_t)  # I've no idea why mpl start at -1...
    ax.pcolormesh(time, yvec, data.T, **kwargs)
    plt.title('%s (%s)' % (lobe, hemi), fontweight='bold')
    plt.yticks(range(n_t), yticks)

    # plt.yticks(range(len(list_rois_)), list_rois_, fontsize=8)

    # remove "R_" and"L_" in yticks labels
    yticks = [y[2:] if y.startswith('R_') or y.startswith('L_') else y
                for y in yticks]
    ax.set_yticklabels(yticks)
    if grid:
        plt.grid()
    if isinstance(vlines, dict):
        for k, kw in vlines.items():
            ax.axvline(k, **kw)
    # Remove xticks
    ax.tick_params(axis='both', which='both', length=0)
    if not xticks:
        plt.tick_params(axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)
    else:
        # Add xticks only for bottom subplots
        if lobe in ['Temporal', 'Occipital']:
            plt.xlabel('Time (s)')

    # Remove borders
    # for k in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[k].set_visible(False)


def da_to_df(da):
    """DataArray to DataFrame conversion."""
    from xarray import DataArray
    assert isinstance(da, DataArray)
    df = da.to_dataframe('df').reset_index()
    df = df.pivot(index='times', columns='roi', values='df')
    return df
