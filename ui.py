import time
from functools                             import partial
from itertools                             import repeat, starmap
import numpy                                                       as np
import pandas                                                      as pd
import torch                                                       as pt
from   PIL                                 import Image
from   unpythonic                          import pipe, piped, exitpipe, singleton, composel, composer, foldl
from   quantiphy                           import Quantity

import hiplot                                                      as hip
import streamlit                                                   as st
from   streamlit                           import runtime
import streamlit.components.v1                                     as components
from   streamlit_extras.add_vertical_space import add_vertical_space
from   annotated_text                      import annotated_text

from   utl                                 import *

# Global RNG Seed
np.random.seed(seed = SEED)
pt.manual_seed(SEED)

# Monkey patching until they merge ...
# https://github.com/facebookresearch/hiplot/pull/274
st._is_running_with_streamlit = runtime.exists()

## Page Formatting and global settings
hide_st_decoration: str = ''' <style> .css-1dp5vir {display: none; visibility: hidden;} </style> '''
with open('resources/about.md') as a:
    about_section = a.read()

## Hamburger
github_url: str        = 'https://github.com/AnonCod3/pcp-demo'
menu_st: dict[str,str] = { 'Get help':     f'{github_url}/discussion'
                         , 'Report a bug': f'{github_url}/issues'
                         , 'About':        about_section }

st.set_page_config( page_title            = 'INDIGO'
                  , page_icon             = ':large_blue_circle:'
                  , layout                = 'wide'
                  , initial_sidebar_state = 'expanded'
                  , menu_items            = menu_st )

## Hide red-yellow gradient
st.markdown(hide_st_decoration, unsafe_allow_html = True)

## Circuit and PDK Selection
ckt = st.sidebar.selectbox('CKT', ckts, index = 0, key = 'ckt')
pdk = st.sidebar.selectbox('PDK', pdks, index = 0, key = 'pdk')
num = st.sidebar.number_input( 'Number of Samples', key = 'num_samples', value = int(10000)
                             , min_value = int(100), max_value = int(1000000))
vdd = st.sidebar.number_input( '$V_{\mathrm{DD}}$ in V', key = 'vdd', value = 2.5
                             , min_value = 1.8, max_value = 3.3, step = 0.3 )
cl  = st.sidebar.number_input( '$C_{\mathrm{L}}$ in pF', key = 'cl', value = 5.0
                             , min_value = 1.0, max_value = 15.0, step = 1.0 )
i0  = st.sidebar.number_input( '$I_{0}$ in μA', key = 'i0', value = 3.0
                             , min_value = 1.0, max_value = 9.0, step = 1.0 )
# tmp = st.sidebar.select_slider( '$T$ in $^{\circ}$C', key = 'temp', value = 27
#                               , options = [-40, 27, 80, 150] )

node_voltages     = [ e for e in elec_params(ckt)
                        if e.endswith('_vds') ]
output_parameters = performance_parameters + node_voltages
design_parameters = [ e for e in elec_params(ckt)
                        if not e.endswith('_vds') ]
sizing_parameters = geom_params(ckt)

nmos, pmos  = update_primitives(pdk)
model_stat  = update_models(pdk, ckt)

def predict(m, x_, ckt_):
    x_params = [ e for e in elec_params(ckt_) if not e.endswith('_vds') ] \
             + testbench_parameters
    y_params = performance_parameters \
             + [ e for e in elec_params(ckt_) if e.endswith('_vds') ]
    x        = x_[x_params].values
    y_       = piped(x) | pt.from_numpy | pt.Tensor.float | m \
                        | pt.Tensor.numpy | exitpipe
    y        = pd.DataFrame( y_.reshape([1,-1]) if len(y_.shape) < 2 else y_
                           , columns = y_params )
    y['fom'] = fom(y.ugbw.values, y.idd.values, cl * 1.0e-12)
    return y

def conf_gen_data(s, n, m, vdd_, i0_, cl_, tmp_ = 27):
    p,c        = n.split('-')
    x          = generate_input_range(p, c, s, i0_, vdd_,)
    x['vdd']   = vdd_
    x['i0']    = i0_ * 1.0e-6
    x['cl']    = cl_ * 1.0e-12
    x['temp']  = tmp_
    y          = predict(m, x, c)
    y['pdk']   = p
    y['ckt']   = c
    return y[performance_parameters + ['pdk', 'ckt', 'fom']]

@st.cache_data
def update_sweep(pdk_, ckt_,num_,vdd_,i0_,cl_,tmp_ = 27):
    design_swp         = generate_input_range(pdk_, ckt_, num_, i0_, vdd_)
    design_swp['vdd']  = vdd_
    design_swp['i0']   = i0_ * 1.0e-6
    design_swp['cl']   = cl_ * 1.0e-12
    design_swp['temp'] = tmp_
    perf_swp           = filter_(predict(model_stat, design_swp, ckt_))
    swp                = design_swp.join(perf_swp)
    szs_swp            = transform( ckt, pdk, nmos, pmos, vdd_, cl_, i0_
                                  , swp[design_parameters + node_voltages] )
    sweep_             = swp.join(szs_swp)
    return output_filter(sweep_)

sweep = update_sweep(pdk, ckt, num, vdd, i0, cl)

@st.cache_resource
def make_experiment( exp_data, compress = True, colorby = None, colormap = None
                   , axes = None, cols = None, n_bins = 100 ) -> hip.Experiment:
    xp             = hip.Experiment.from_dataframe(exp_data)
    if colorby and colormap:
        xp.colorby = colorby
        xp.parameters_definition[colorby].colormap = colormap
    xp._compress   = compress
    axes_shown     = axes if axes else list(exp_data.columns)
    axes_hidden    = ['from_uid', 'uid'] \
                   + ([c for c in exp_data.columns if c not in axes] if axes else [])
    xp.display_data( hip.Displays.PARALLEL_PLOT
                   ).update({ 'order': axes_shown, 'hide': axes_hidden })
    cols_shown     = ['uid'] + (cols if cols else list(exp_data.columns))
    cols_hidden    = ['from_uid'] \
                   + ([ c for c in exp_data.columns if c not in cols] if cols else [])
    xp.display_data( hip.Displays.TABLE
                   ).update({ 'order': cols_shown, 'hide':  cols_hidden })
    xp.display_data(hip.Displays.DISTRIBUTION).update({'nbins': n_bins})
    xp.display_data( hip.Displays.XY
                   ).update({ 'dots_opacity':              1.0
                            , 'dots_thickness':            3.0
                            , 'dots_highlighed_thickness': 2.5 })
    return xp

## Warning
if num > 20000:
    st.warning( 'Using more than 20000 points may cause lag, due to rendering'
              , icon = '⚠️' )

## Warning
st.warning( 'App may lag due to memory restrictions of public hosting space'
    , icon = '⚠️' )

## Tabs
tab_names = [ 'Performance Space', 'Design Space'
            , 'Circuit Confusion', '$g_{\mathrm{m}}/I_{\mathrm{d}}$'
            , 'Schematic', 'Legend', 'README' ]
tab_perf, tab_desgn, tab_conf, tab_gmid, tab_scm, tab_leg, tab_rm = st.tabs(tab_names)

with tab_perf:
    st.write('# Performance Space Exploration')
    axes = ['fom'] + performance_parameters
    xpp = make_experiment( sweep
                         , colorby  = 'fom'
                         , axes     = list(reversed(axes))
                         , cols     = axes + design_parameters + sizing_parameters)

    for p in axes:
        if p in ['ugbw', 'sr_r', 'voff_stat']:
            xpp.parameters_definition[p].type = hip.ValueType.NUMERIC_LOG

    xpp.to_streamlit(ret = 'selected_uids', key = 'xpp').display()

with tab_desgn:
    st.write('# Design Space Exploration')
    dps_select = st.multiselect( 'Design Parameters', design_parameters
                               , default = ('MNDP11_gmoverid', 'MNDP11_fug') )
    sps_select = st.multiselect( 'Sizing Parameters', sizing_parameters
                               , default = ('Wdp1', 'Ldp1') )
    pps_select = st.multiselect( 'Performance Parameters', performance_parameters + ['fom']
                               , default = ('a_0', 'pm') )

    design_axes = dps_select + sps_select + pps_select

    xpd  = make_experiment( sweep
                          , colorby  = pps_select[0]
                          , axes     = design_axes
                          , cols     = design_axes )

    for p in design_axes:
        dc = p.endswith('_fug') or p.startswith('W') or p.startswith('L')
        pc = p in ['ugbw', 'sr_r']
        if dc or pc:
            xpd.parameters_definition[p].type = hip.ValueType.NUMERIC_LOG

    xpd.to_streamlit(ret = 'selected_uids', key = 'xpd').display()

## Confusions
with tab_conf:
    st.write('# Confusions')
    ckts_sel = st.multiselect( 'Select **CKT**s for confusion', ckts
                             , default=('sym', 'rfa', 'mil') )
    pdks_sel = st.multiselect( 'Select **PDK**s for confusion', pdks
                             , default = ('90nm', '45nm') )
    s        = st.number_input( 'Samples per Model', value = int(1000)
                              , min_value = int(10) , max_value = int(100000) )

    if (len(ckts_sel) > 0) and (len(pdks_sel) > 0):
        conf_models = { f'{p}-{c}': pt.jit.load(f'./models/{p}/{c}-stat.pt')
                        for c in ckts_sel for p in pdks_sel }
        conf_data   = piped( [ conf_gen_data(s, n, m, vdd, i0, cl)
                               for n,m in conf_models.items() ]
                           ) | pd.concat | output_filter | exitpipe

        xpc = make_experiment( conf_data
                             , colorby  = 'ckt' if len(ckts_sel) > len(pdks_sel) else 'pdk'
                             , colormap = 'schemeSet3'
                             , axes     = ['pdk', 'ckt', 'fom'] + performance_parameters
                             , cols     = ['pdk', 'ckt', 'fom'] + performance_parameters )

        xpc.parameters_definition['ckt'].type = hip.ValueType.CATEGORICAL
        xpc.parameters_definition['pdk'].type = hip.ValueType.CATEGORICAL

        xpc.to_streamlit(ret = 'selected_uids', key = 'xpc').display()
    else:
        st.write('Select $\geq 1$ **CKT** and $\geq 1$ **PDK** from the list above.')

with tab_gmid:
    st.write('# $\\frac{g_{\mathrm{m}}}{I_{\mathrm{D}}}$ Model Exploration')
    ib_l, ib_h = st.slider( 'Drain Current $| I_{\mathrm{D}} |$ in μA'
                          , min_value = 0.1, max_value = 10.0
                          , value = (1.0, 6.0), step = 0.1 )

    mos_cols   = ['id', 'gmoverid', 'fug', 'vds', 'vbs', 'W', 'L', 'vgs']
    nx         = generate_primitive_range(pdk, num, 'n', ib_l, ib_h)
    ny         = mos_predict(nmos, nx)
    sweep_nmos = pd.DataFrame(np.concatenate([nx, ny], axis = 1), columns = mos_cols)
    px         = generate_primitive_range(pdk, num, 'p', ib_l, ib_h)
    py         = mos_predict(pmos, px)
    sweep_pmos = pd.DataFrame(np.concatenate([px, py], axis = 1), columns = mos_cols)

    st.write('### NMOS')
    xp_nmos = make_experiment( sweep_nmos
                             , colorby  = 'gmoverid'
                             , axes     = mos_cols
                             , cols     = mos_cols )

    xp_nmos.parameters_definition['W'].type   = hip.ValueType.NUMERIC_LOG
    xp_nmos.parameters_definition['L'].type   = hip.ValueType.NUMERIC_LOG
    xp_nmos.parameters_definition['fug'].type = hip.ValueType.NUMERIC_LOG

    xp_nmos.to_streamlit(key = 'xp_nmos').display()

    st.write('### PMOS')
    xp_pmos = make_experiment( sweep_pmos
                             , colorby  = 'gmoverid'
                             , axes     = mos_cols
                             , cols     = mos_cols )

    xp_pmos.parameters_definition['W'].type   = hip.ValueType.NUMERIC_LOG
    xp_pmos.parameters_definition['L'].type   = hip.ValueType.NUMERIC_LOG
    xp_pmos.parameters_definition['fug'].type = hip.ValueType.NUMERIC_LOG

    xp_pmos.to_streamlit(key = 'xp_pmos').display()

## Schematic View
with tab_scm:
    st.image( Image.open(f'resources/{ckt}.png')
            , caption = f'Schematic: {ckt}' )

## Schematic Legend
with tab_leg:
    header = '''
| Symbol | Description | Unit | Axis/Column ID |
|--------|-------------|------|----------------|'''

    performance_desc  = [ 'DC Loop Gain'
                        , 'Unity Gain Bandwidth'
                        , 'Phase Margin'
                        , 'Gain Margin'
                        , 'Slew Rate (rising)'
                        , 'Power Supply Rejection Ratio (positiv)'
                        , 'Power Supply Rejection Ratio (negative)'
                        , 'Common Mode Rejection Ratio'
                        , 'Slew Rate (falling)'
                        , 'Statistical Offset'
                        , 'Current Consumption'
                        , 'Output Voltage Swing (high)'
                        , 'Output Voltage Swing (low)'
                        , 'Figure of Merit: $\mathrm{UGBW} · C_{\mathrm{L}} / I_{\mathrm{DD}}$ ']
    perf_rows         = starmap( lambda s,d,u,p: f'| ${s}$ | {d} | {u} | `{p}` |'
                               , zip( performance_syms
                                    , performance_desc
                                    , performance_units
                                    , performance_parameters + ['fom']))

    def elec_row(p: str):
        i,t = p.split('_')
        s = { 'fug':      f'f_{{\mathrm{{ug}},\mathtt{{{i}}}}}'
            , 'gmoverid': f'\\frac{{g_{{\mathrm{{m}}}}}}{{I_{{\mathrm{{d}}}}}}_{{\mathtt{{{i}}}}}'
            , 'id':       f'I_{{\mathrm{{d}},\mathtt{{{i}}}}}'
            }.get(t, 'N/A')
        u = { 'fug': 'Hz', 'gmoverid': '1/V', 'id': 'A' }.get(t, 'N/A')
        d = { 'fug':      f'Speed for `{i}`'
            , 'gmoverid': f'Efficiency for `{i}`'
            , 'id':       f'Drain Current for `{i}`'
            }.get(t, 'N/A')
        return f'| ${s}$ | {d} | {u} | `{p}` |'
    elec_rows = map(elec_row, design_parameters)

    def geom_row(p: str):
        t = p[0]
        i = p[1:].upper()
        j = list({d.split('_')[0] for d in design_parameters if i in d})
        i_ = (j[0] if j else f'MX{i}') if t == 'M' else \
             ((j[0][:-1] + '*') if j else (f'MX{i}' + '*'))
        s = f'{t}_{{\mathtt{{{i}}}}}'
        u = '-' if t == 'M' else 'm'
        d = { 'W': f'Channel Width for `{i_}`'
            , 'L': f'Channel Length for `{i_}`'
            , 'M': f'Multiplier for `{i_}`'
            }.get(t, 'N/A')
        return f'| ${s}$ | {d} | {u} | `{p}` |'
    geom_rows = map(geom_row, sizing_parameters)

    testbench_units      = ['V', 'A', 'F', '°C']
    testbench_syms       = ['V_{\mathrm{DD}}', 'I_{0}', 'C_{\mathrm{L}}', 'T']
    testbench_desc       = [ 'Supply Voltage', 'Bias Current'
                           , 'Load Capacitance', 'Temperature' ]
    tebe_rows            = starmap( lambda s,d,u,p: f'| ${s}$ | {d} | {u} | `{p}` |'
                                  , zip( testbench_syms
                                       , testbench_desc
                                       , testbench_units
                                       , testbench_parameters ))

    primitive_parameters = ['id', 'gmoverid', 'fug', 'vds', 'vbs', 'W', 'L']
    primitive_units      = ['A', '1/V', 'Hz', 'V', 'V', 'm', 'm']
    primitive_syms       = [ 'I_{\mathrm{d}}', '\\frac{g_{\mathrm{m}}}{I_{\mathrm{d}}}'
                           , 'f_{\mathrm{ug}}', 'V_{\mathrm{ds}}', 'V_{\mathrm{bs}}'
                           , 'W', 'L' ]
    primitive_desc       = [ 'Drain Current', 'Efficiency (transconductance over drain current)'
                           , 'Speed (unity gain frequency)'
                           , 'Drain-Source Voltage', 'Bulk-Source Voltage'
                           , 'Channel Width', 'Channel Length' ]
    prim_rows            = starmap( lambda s,d,u,p: f'| ${s}$ | {d} | {u} | `{p}` |'
                                  , zip( primitive_syms
                                       , primitive_desc
                                       , primitive_units
                                       , primitive_parameters ))

    st.write('## Performance Parameters')
    st.write('\n'.join([header] + list(perf_rows)))
    st.write('## Testbench Parameters')
    st.write('\n'.join([header] + list(tebe_rows)))
    st.write('## Electrical Design Parameters')
    st.write('\n'.join([header] + list(elec_rows)))
    st.write('## Geometrical Design Parameters')
    st.write('\n'.join([header] + list(geom_rows)))
    st.write('## Device Model Parameters')
    st.write('\n'.join([header] + list(prim_rows)))
    
with tab_rm:
    with open('README.md') as r:
        readme_section = r.read()
        
    st.markdown(readme_section, unsafe_allow_html = True)

