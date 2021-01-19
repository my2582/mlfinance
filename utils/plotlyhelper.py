import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
import cufflinks as cf

from itertools import chain

import copy

def plotly_fig2json(fig, fpath=None):
    """
    Serialize a plotly figure object to JSON so it can be persisted to disk.
    Figure's persisted as JSON can be rebuilt using the plotly JSON chart API:

    http://help.plot.ly/json-chart-schema/

    If `fpath` is provided, JSON is written to file.

    Modified from https://github.com/nteract/nteract/issues/1229
    Source: https://github.com/plotly/plotly.py/issues/579
    """

    redata = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    relayout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    fig_json=json.dumps({'data': redata,'layout': relayout})

    if fpath:
        with open(fpath, 'w') as f:
            f.write(fig_json)
    else:
        return fig_json

def plotly_from_json(fpath):
    """Render a plotly figure from a json file"""
    with open(fpath, 'r') as f:
        v = json.loads(f.read())

    fig = go.Figure(data=v['data'], layout=v['layout'])
    iplot(fig, show_link=False)


def plotly_multi_shades(fig, x0, x1, color='gray'):
    """ Adds shaded areas for specified dates in a plotly plot.
        The lines of the areas are set to transparent using rgba(0,0,0,0)

    Examples
    --------
        rec_fig = cli.iplot(
            asFigure=True,
            kind='scatter',
            xTitle='Dates',
            yTitle='Returns',
            title='OECD CLI USA with recessions in gray',
            colors='blue',

            # We need `vspan` specified.
            vspan={
                'x0': rec_starts,
                'x1': rec_ends,
                'color': 'gray',
                'fill': True,
                'opacity': .4
            }
        )
        rec_fig = multiShades(rec_fig, x0=rec_starts, x1=rec_ends)
        iplot(rec_fig)

        >> rec_starts
            ['1957-09-06', '1960-05-06', '1970-01-02']
    """

    # get dict from tuple made by vspan()
    xElem = fig['layout']['shapes'][0]

    # container (list) for dicts / shapes
    shp_lst=[]

    # make dicts according to x0 and X1
    # and edit elements of those dicts
    for i in range(0,len(x0)):
        shp_lst.append(copy.deepcopy(xElem))
        shp_lst[i]['x0'] = x0[i]
        shp_lst[i]['x1'] = x1[i]
        shp_lst[i]['line']['color'] = 'rgba(0,0,0,0)'

    # replace shape in fig with multiple new shapes
    fig['layout']['shapes']= tuple(shp_lst)

    return fig

def plotly_multi_shades(fig, x0, x1, colors=['gray'], alpha=[0.2]):
    """ Adds shaded areas for specified dates in a plotly plot.
        The lines of the areas are set to transparent using rgba(0,0,0,0).
        This function allows a list of lists for x0 and x1 for which
        different colors may be filled with opacity of alpha.

    Examples
    --------
        fig = plotly_multi_shades(fig, x0=[rec_starts, growth_starts],
                                x1=[rec_ends, growth_ends],
                                colors=['gray', 'green'], alpha=0.2)
        iplot(fig)
    """
    if len(x0) != len(x1):
        raise ValueError('x0 and x1 must have the same length.')
    
    shp_tuples = []
    for i in range(len(x0)):
        dt0 = x0[i]
        dt1 = x1[i]
        shp_list = []
        for j in range(len(dt0)):
            shp_list.append(dict(type='rect', xref='x', yref='paper', y0=0, y1=1))
            shp_list[j]['x0']=dt0[j]
            shp_list[j]['x1']=dt1[j]
            alp = alpha[i] if isinstance(alpha, list) else alpha
            shp_list[j]['fillcolor'] = cf.colors.to_rgba(colors[i], alp)
            shp_list[j]['line'] = {'color': 'rgba(0,0,0,0)', 'dash': 'solid', 'width': 1}

        shp_tuples.append(tuple(shp_list))

    shp = tuple(chain.from_iterable(shp_tuples))
    fig['layout'].update(shapes=shp)
    
    return fig