import os
from   itertools          import repeat
import torch                                as pt
import numpy                                as np
from   scipy.stats        import qmc
from   scipy.optimize     import minimize
import pandas                               as pd
import streamlit                            as st
from   annotated_text     import annotation
import yaml
from   apm                import *
from   functools          import partial
from   unpythonic         import pipe, piped, exitpipe, first

## Global RNG Seed
SEED: int = 666

np.random.seed(seed = SEED)
pt.manual_seed(SEED)

## Available circuits and PDKs
ckts: tuple[str] = ('sym', 'mil', 'rfa')
pdks: tuple[str] = ('90nm', '45nm')

## Serafin Path
serafin_home = os.path.expanduser('~/.serafin')

## Colors
green, red, yellow, blue, white = '#87d276', '#ff9494', '#F6BD60', '#00387b', '#F7F7F7'

## Floating Point number formatter for DataFrames
def formatter(x):
    return '{:,.2E}'.format(x)

def button_txt(txt: str):
    return annotation(txt, background = blue, color = white)

# Calculate FOM
def fom(ugbw: np.array, idd: np.array, cl: np.array) -> np.array:
    return ((ugbw * cl) / idd) * 1000

## Available Performance Parameters that can be predicted by a Model (in this order)
performance_parameters = [ 'a_0', 'ugbw', 'pm', 'gm', 'sr_r', 'psrr_p', 'psrr_n'
                         , 'cmrr', 'sr_f', 'voff_stat', 'idd', 'v_oh', 'v_ol' ]
performance_units      = [ 'dB', 'Hz', '°', 'dB', 'V/μs', 'dB', 'dB'
                         , 'dB', 'V/μs', 'V', 'A', 'V', 'V', 'MHz pF / mA' ]
performance_syms       = [ 'A_0', '\mathrm{UGBW}', '\mathrm{PM}', '\mathrm{GM}'
                         , '\mathrm{SR}_{\mathrm{r}}', '\mathrm{PSRR}_{\mathrm{p}}'
                         , '\mathrm{PSRR}_{\mathrm{n}}', '\mathrm{CMRR}'
                         , '\mathrm{SR}_{\mathrm{f}}', 'V_{\mathrm{off}}(1σ)'
                         , 'I_{\mathrm{DD}}', 'V_{\mathrm{oh}}', 'V_{\mathrm{ol}}'
                         , '\mathrm{FOM}' ]
performance_predicate  = [ True, True, True, False, True, True, True
                         , True, True, False, False, False, False ]
## Testbench Settings
testbench_parameters = ['vdd', 'i0', 'cl', 'temp']

## Performance Styler
def styler_perf(sim_, prd_):
    prd = prd_.values
    sim = sim_.values
    cnd = np.array( [[float(p) for p in performance_predicate]]
                  ).repeat(prd.shape[0], axis = 0)
    suc = ((cnd * (sim > prd)) + ((1.0 - cnd) * (sim < prd)))
    cls = (np.abs((sim - prd) / prd) < 0.1)
    return np.where( suc, f'background-color: {green}'
                   , np.where( cls, f'background-color: {yellow}'
                                  , f'background-color: {red}' ))

def styler_op(sim_, prd_):
    prd = prd_.values
    sim = sim_.values
    suc = (np.abs((sim - prd) / prd) < 0.05)
    cls = (np.abs((sim - prd) / prd) < 0.1)
    return np.where( suc, f'background-color: {green}'
                   , np.where( cls, f'background-color: {yellow}'
                                  , f'background-color: {red}' ))

def styler_neutral(df: pd.DataFrame) -> np.array:
    return np.full_like(df.values, f'background-color: {white}', dtype=object)

#@st.cache_resource
def update_primitives(pdk: str):
    nmos = pt.jit.load(f'./models/{pdk}/nmos.pt')
    pmos = pt.jit.load(f'./models/{pdk}/pmos.pt')
    return (nmos,pmos)

#@st.cache_resource
def update_models(pdk: str, ckt: str):
    model_stat = pt.jit.load(f'./models/{pdk}/{ckt}-stat.pt')
    return model_stat

## Read default testbench parameters
def defaults(pdk: str, ckt: str, key: str = None):
    with open(f'./resources/ckt/{ckt}.yml', 'r') as yml:
        ckt_cfg = yaml.safe_load(yml)
    with open(f'./resources/pdk/{pdk}.yml', 'r') as yml:
        pdk_cfg = yaml.safe_load(yml)
    cfg = pdk_cfg['testbench'] | ckt_cfg['parameters']['testbench']
    return cfg[key] if key else cfg

## Geometrical Sizing Parameter Names for a given Circuit
def geom_params(ckt: str) -> list[str]:
    with open(f'./resources/ckt/{ckt}.yml', 'r') as yml:
        cfg = yaml.safe_load(yml)
    return list(cfg['parameters']['geometrical'].keys())

@case_distinction
def optim_geom_params(ckt: Match(Strict('sym'))) -> list[str]:
    return [ 'Ldp1',  'Lcm1',  'Lcm3',  'Lcm4'
           , 'Wdp1',  'Wcm1',  'Wcm3',  'Wcm4' ]
@case_distinction
def optim_geom_params(ckt: Match(_)) -> [str]:
    raise ValueError(f'optim_geom_params not defined for {ckt}')

@case_distinction
def generalize_sizing(ckt: Match(Strict('sym')), sizing: pd.DataFrame):
    sizing['Lcm2']  = sizing['Lcm3']
    sizing['Wcm2']  = sizing['Wcm3']
    return sizing
@case_distinction
def generalize_sizing(ckt: Match(_), sizing: Match(_)) -> [str]:
    raise ValueError(f'generalize_sizing not defined for {ckt}')

## Electrical Design Parameters of a given Circuit
@case_distinction
def elec_params(ckt: Match(Strict('sym'))) -> [str]:
    return [ 'MNDP11_gmoverid', 'MNCM11_gmoverid', 'MPCM31_gmoverid', 'MNCM41_gmoverid'
           , 'MNDP11_fug', 'MNCM11_fug', 'MPCM31_fug', 'MNCM41_fug'
           , 'MNCM12_id', 'MNCM42_id'
           , 'MNDP11_vds', 'MNCM11_vds', 'MPCM31_vds', 'MNCM41_vds' ]
@case_distinction
def elec_params(ckt: Match(Strict('mil'))) -> [str]:
    return [ 'MNDP11_gmoverid', 'MNCM11_gmoverid', 'MPCM21_gmoverid', 'MPCS11_gmoverid'
           , 'MNDP11_fug', 'MNCM11_fug', 'MPCM21_fug', 'MPCS11_fug'
           , 'MNCM12_id', 'MNCM13_id'
           , 'MNDP11_vds', 'MNCM11_vds', 'MPCM21_vds', 'MPCS11_vds' ]
@case_distinction
def elec_params(ckt: Match(Strict('rfa'))) -> [str]:
    return [ 'MNDP11_gmoverid', 'MPDP21_gmoverid', 'MNCM11_gmoverid', 'MPCM21_gmoverid'
           , 'MNCM31_gmoverid', 'MNLS11_gmoverid', 'MPLS21_gmoverid'
           , 'MNDP11_fug', 'MPDP21_fug', 'MNCM11_fug', 'MPCM21_fug'
           , 'MNCM31_fug', 'MNLS11_fug', 'MPLS21_fug'
           , 'MNCM13_id',  'MNCM32_id'
           , 'MNDP11_vds', 'MPDP21_vds', 'MNCM11_vds' , 'MPCM21_vds'
           , 'MNCM31_vds', 'MNLS11_vds', 'MPLS21_vds' ]
@case_distinction
def elec_params(ckt: Match(_)) -> [str]:
    raise ValueError(f'elec_params not defined for {ckt}')

@case_distinction
def ib_range(ckt: Match(Strict('sym'))):
    return [(1,4), (3,9)]
@case_distinction
def ib_range(ckt: Match(Strict('mil'))):
    return [(1,9), (10,41)]
@case_distinction
def ib_range(ckt: Match(Strict('rfa'))):
    return [(1,4), (2,5)]
@case_distinction
def ib_range(ckt: Match(_)):
    raise ValueError(f'No branch currents for {ckt} defined')

def rng_currents( ibs: list[tuple[int,int]], num:int, i0: float
                          ) -> [np.array]:
    return [ np.random.randint(l, h, size = (num,1)).astype(float) * i0
             for l,h in ibs ]

def filter_(df: pd.DataFrame) -> pd.DataFrame:
    return df[( (df.pm > 0) & (df.gm < 0)
              & (df.sr_r > 1.0e5) & (df.sr_f < -1.0e5)
              & (df.voff_stat > 0.0) )]

@st.cache_data
def generate_input_range(pdk: str, ckt: str, num: int, i0: float, vdd: float):
    d_params = [ p for p in elec_params(ckt)
                 if (not p.endswith('_vds')) and (not p.endswith('_id'))]
    lb       = [ 5.0  if p.endswith('_gmoverid') else 6.5 for p in d_params ]
    ub       = [ 20.0 if p.endswith('_gmoverid') else 9.0 for p in d_params ]
    msk      = np.array([p.endswith('_fug') for p in d_params])
    lhs      = qmc.LatinHypercube(d = len(d_params), seed = SEED)
    smp      = qmc.scale(lhs.random(n = num), lb, ub)
    sample   = np.where(msk, np.power(10, smp), smp)
    d_df     = pd.DataFrame(sample, columns = d_params)
    i_params = [p for p in elec_params(ckt) if p.endswith('_id')]
    i_df     = piped(ckt) | ib_range \
                          | partial(rng_currents, num = num, i0 = i0 * 1.0e-6) \
                          | partial(np.concatenate, axis = 1) \
                          | partial(pd.DataFrame, columns = i_params) | exitpipe
    return d_df.join(i_df)

@st.cache_data
def generate_primitive_range( pdk: str, num: int, mos_type: str = 'n'
                            , ib_l: float = 1.0, ib_h: float = 6.0 ):
    lb     = [ ib_l * 1.0e-6, 5.0, 6.0, 0.2, -1.5 ] if mos_type == 'n' else \
             [ -1.0e-4, 5.0, 6.0, -1.8, 0.0 ]
    ub     = [ ib_h * 1.0e-6, 20.0, 10.0, 1.8, 0.0 ] if mos_type == 'n' else \
             [ -1.0e-8, 20.0, 10.0, -0.2, 1.5 ]
    msk    = np.array([False, False, True, False, False])
    lhs    = qmc.LatinHypercube(d = 5, seed = SEED)
    smp    = qmc.scale(lhs.random(n = num), lb, ub)
    sample = np.where(msk, np.power(10, smp), smp)
    return sample

def output_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[ (df.gm   < 0.0)
             & (df.pm   > 0.0)
             & (df.sr_f < 0.0)
             & (df.sr_r > 0.0)
             ].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True)

def mos_inputs(prefix: str) -> list[str]:
    params = ['gmoverid', 'fug', 'vds']
    return [f'{prefix}_{p}' for p in params ]

## Primitive Device Prediction
def mos_predict(mdl, data):
    return piped(data) | pt.from_numpy | pt.Tensor.float | mdl | pt.Tensor.numpy | exitpipe

## Capacitance calculation
@case_distinction
def cap_predict(pdk: Match(Strict('180nm')), c):
    w = piped(c) | partial(np.multiply , 1.0e15) | np.sqrt | partial(np.multiply, 1.0e-6) \
                 | partial(np.reshape, newshape = [-1,1]) | exitpipe
    return np.concatenate([ w, w ], axis = 1)
@case_distinction
def cap_predict(pdk: Match(Strict('90nm')), c):
    w = piped(c) | partial(np.multiply , 1.0e15) | np.sqrt | partial(np.multiply, 1.0e-6) \
                 | partial(np.reshape, newshape = [-1,1]) | exitpipe
    return np.concatenate([ w, w ], axis = 1)
@case_distinction
def cap_predict(pdk: Match(Strict('45nm')), c):
    w = piped(c) | partial(np.multiply , 1.1e15) | np.sqrt | partial(np.multiply, 1.0e-6) \
                 | partial(np.reshape, newshape = [-1,1]) | exitpipe
    return np.concatenate([ w, w ], axis = 1)
@case_distinction
def cap_predict(pdk: Match(_), c):
    raise ValueError(f'No Capcitance defined for {pdk}')

## Resistance calculation
@case_distinction
def res_predict(pdk: Match(Strict('180nm')), r):
    w = np.ones_like(r) * 540.0e-9
    l = (r * w) / 7.5
    return np.concatenate([ w.reshape([-1,1]), l.reshape([-1,1]) ], axis = 1)
@case_distinction
def res_predict(pdk: Match(Strict('90nm')), r):
    w = np.ones_like(r) * 540.0e-9
    l = (r * w) / 0.5
    return np.concatenate([ w.reshape([-1,1]), l.reshape([-1,1]) ], axis = 1)
@case_distinction
def res_predict(pdk: Match(Strict('45nm')), r):
    w = np.ones_like(r) * 540.0e-9
    l = (r * w) / 0.08
    return np.concatenate([ w.reshape([-1,1]), l.reshape([-1,1]) ], axis = 1)
@case_distinction
def res_predict(pdk: Match(_), data):
    raise ValueError(f'No Resistance defined for {pdk}')

@case_distinction
def transform( ckt: Match(Strict('sym')), pdk: Match(_)
             , nmos: Match(_), pmos: Match(_)
             , vdd: Match(_), cl: Match(_), i0: Match(_), df: Match(_)
             ) -> pd.DataFrame:
    num    = df.shape[0]
    vdd    = np.ones(num) * vdd
    i0     = np.ones(num) * i0 * 1.0e-6
    vss    = np.zeros(num)
    vcm    = (np.abs(df.MPCM31_vds.values) + np.abs(df.MNDP11_vds.values)) - vdd
    i1_    = np.abs(df.MNCM12_id.values)
    i2_    = np.abs(df.MNCM42_id.values)
    Mcm11  = np.ones(num)
    Mcm12  = piped(i1_ / i0) | np.round | partial(np.maximum, 1.0) | exitpipe
    Mcm21  = Mcm31 = np.ones(num)
    Mcm22  = Mcm32 = piped((2.0 * i2_) / i1_) | np.round | partial(np.maximum, 1.0) | exitpipe
    Mcm41  = Mcm42 = np.ones(num) * 2.0
    Mdp11  = Mdp12 = np.ones(num) * 2.0
    i1     = i0 * Mcm12 / Mcm11
    i12    = i1 / 2.0
    i2     = i12 * Mcm32 / Mcm31
    cm1_in = np.concatenate( [ i0.reshape([-1,1])
                             , df[mos_inputs('MNCM11')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1 )
    cm3_in = np.concatenate( [ i12.reshape([-1,1])
                             , df[mos_inputs('MPCM31')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1 )
    cm4_in = np.concatenate( [ i2.reshape([-1,1])
                             , df[mos_inputs('MNCM41')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1 )
    dp1_in = np.concatenate( [ i12.reshape([-1,1])
                             , df[mos_inputs('MNDP11')].values
                             , vcm.reshape([-1,1]) ]
                           , axis = 1 )
    cm1    = mos_predict(nmos, cm1_in)
    cm3    = mos_predict(pmos, cm3_in)
    cm4    = mos_predict(nmos, cm4_in)
    dp1    = mos_predict(nmos, dp1_in)
    Wcm1   = cm1[:,0] / Mcm11
    Wcm2   = Wcm3 = cm3[:,0] / Mcm21
    Wcm4   = cm4[:,0] / Mcm41
    Wdp1   = dp1[:,0] / Mdp11
    Lcm1   = cm1[:,1]
    Lcm2   = Lcm3 = cm3[:,1]
    Lcm4   = cm4[:,1]
    Ldp1   = dp1[:,1]
    sizing = np.concatenate( [ s.reshape([-1,1])
                               for s in [ Ldp1,  Lcm1,  Lcm2,  Lcm3,  Lcm4
                                        , Wdp1,  Wcm1,  Wcm2,  Wcm3,  Wcm4
                                        , Mdp11, Mdp12, Mcm11, Mcm12, Mcm21
                                        , Mcm22, Mcm31, Mcm32, Mcm41, Mcm42 ] ]
                            , axis = 1 )
    return pd.DataFrame(sizing, columns = geom_params('sym'))

@case_distinction
def transform( ckt: Match(Strict('mil')), pdk: Match(_)
             , nmos: Match(_), pmos: Match(_)
             , vdd: Match(_), cl: Match(_), i0: Match(_), df: Match(_)
             ) -> pd.DataFrame:
    num    = df.shape[0]
    vdd    = np.ones(num) * vdd
    vss    = np.zeros(num)
    cl     = np.ones(num) * cl * 1.0e-12
    i0     = np.ones(num) * i0 * 1.0e-6
    i1_    = np.abs(df.MNCM12_id.values)
    i2_    = np.abs(df.MNCM13_id.values)
    Mcap1  = np.ones(num)
    Mres1  = np.ones(num)
    Mcm11  = np.ones(num)
    Mcm12  = piped(i1_ / i0) | np.round | partial(np.maximum, 1.0) | exitpipe
    Mcm13  = piped(i2_ / i0) | np.round | partial(np.maximum, 1.0) | exitpipe
    Mcs11  = Mcm13
    Mdp12  = Mdp11 = np.ones(num) * 2.0
    Mcm22  = Mcm21 = np.ones(num) * 2.0
    i1     = i0 * Mcm12
    i2     = i0 * Mcm13
    i12    = i1 / 2.0
    vcm    = np.abs(df.MNDP11_vds.values) + np.abs(df.MPCM21_vds.values) - vdd
    cm1_in = np.concatenate( [ i0.reshape([-1,1])
                             , df[mos_inputs('MNCM11')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1)
    cm2_in = np.concatenate( [ i12.reshape([-1,1])
                             , df[mos_inputs('MPCM21')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1)
    cs1_in = np.concatenate( [ i2.reshape([-1,1])
                             , df[mos_inputs('MPCS11')].values
                             , vss.reshape([-1,1]) ]
                           , axis = 1)
    dp1_in = np.concatenate( [ i12.reshape([-1,1])
                             , df[mos_inputs('MNDP11')].values
                             , vcm.reshape([-1,1]) ]
                           , axis = 1)
    cp1_in = i2 / 3.25e6
    rs1_in = (1.0 / (df.MPCS11_gmoverid.values * i2)) * ((cp1_in + cl) / cp1_in)
    cm1    = mos_predict(nmos, cm1_in)
    cm2    = mos_predict(pmos, cm2_in)
    cs1    = mos_predict(pmos, cs1_in)
    dp1    = mos_predict(nmos, dp1_in)
    cp1    = cap_predict(pdk, cp1_in)
    rs1    = res_predict(pdk, rs1_in)
    Wcm1   = cm1[:,0] / Mcm11
    Wcm2   = cm2[:,0] / Mcm21
    Wcs1   = cs1[:,0] / Mcs11
    Wdp1   = dp1[:,0] / Mdp11
    Wdp1   = dp1[:,0] / Mdp11
    Wcap   = cp1[:,0] / Mcap1
    Wres   = rs1[:,0] / Mres1
    Lcm1   = cm1[:,1]
    Lcm2   = cm2[:,1]
    Lcs1   = cs1[:,1]
    Ldp1   = dp1[:,1]
    Lcap   = cp1[:,1]
    Lres   = rs1[:,1]
    sizing = np.concatenate( [ s.reshape([-1,1])
                               for s in [ Lcm1, Lcm2, Ldp1, Lcs1, Lcap, Lres
                                        , Wcm1, Wcm2, Wdp1, Wcs1, Wcap, Wres
                                        , Mcm11, Mcm12, Mcm13, Mcm21, Mcm22
                                        , Mdp11, Mdp12, Mcs11, Mcap1, Mres1 ] ]
                            , axis = 1)
    return pd.DataFrame(sizing, columns = geom_params('mil'))

@case_distinction
def transform( ckt: Match(Strict('rfa')), pdk: Match(_)
             , nmos: Match(_), pmos: Match(_)
             , vdd: Match(_), cl: Match(_), i0: Match(_), df: Match(_)
             ) -> pd.DataFrame:
    num     = df.shape[0]
    vdd     = np.ones(num) * vdd
    vss     = np.zeros(num)
    i0      = np.ones(num) * i0 * 1.0e-6
    i3      = i0
    i4      = i0
    i5      = i0
    i1_     = np.abs(df.MNCM13_id.values)
    i2_     = i1_
    i6_     = np.abs(df.MNCM32_id.values)
    Mcm14   = Mcm12 = Mcm11 = np.ones(num)
    Mcm13   = piped(i1_ / i0) | np.round | partial(np.maximum, 1.0) \
            | partial(np.multiply, Mcm11) | exitpipe
    Mcm23   = Mcm21 = np.ones(num)
    Mcm22   = piped(i2_ / i3) | np.round | partial(np.maximum, 1.0) \
            | partial(np.multiply, Mcm21) | exitpipe
    Mcm24   = piped(i6_ / i3) | np.round | partial(np.maximum, 1.0) \
            | partial(np.multiply, Mcm21) | exitpipe
    Mcm25   = Mcm24
    Mcm32   = Mcm31 = np.ones(num) * 2.0
    Mls12   = Mls11 = np.ones(num)
    Mls22   = Mls21 = np.ones(num)
    Mdp12   = Mdp11 = np.ones(num) * 2.0
    Mdp22   = Mdp21 = np.ones(num) * 2.0
    Mrf21   = Mrf11 = np.ones(num)
    i1      = (i0 * Mcm13 / Mcm11) - 0.5e-6
    i2      = (i3 * Mcm22) / Mcm21
    i12     = i1 / 2.0
    i22     = i2 / 2.0
    i8      = i6 = (i3 * Mcm24) / Mcm21
    i7      = (i6 - i22) + (i8 -i12)
    vw      = np.abs(df.MPLS21_vds.values) + np.abs(df.MNLS11_vds.values) + np.abs(df.MNCM31_vds.values)
    vy      = np.abs(df.MNCM31_vds.values)
    vcm1    = np.abs(df.MNDP11_vds.values) - vw
    vcm2    = np.abs(df.MNCM31_vds.values) + np.abs(df.MPDP21_vds.values)
    gmid_rf = np.ones(num) * 5.0
    fug_rf  = np.ones(num) * (10.0 ** 9.1)
    vds_rf  = np.ones(num) * (vdd / 2.0)
    cm1_in  = np.concatenate( [ i0.reshape([-1,1])
                              , df[mos_inputs('MNCM11')].values
                              , vss.reshape([-1,1]) ]
                            , axis = 1)
    cm2_in  = np.concatenate( [ i3.reshape([-1,1])
                              , df[mos_inputs('MPCM21')].values
                              , vss.reshape([-1,1]) ]
                            , axis = 1)
    cm3_in  = np.concatenate( [ i8.reshape([-1,1])
                              , df[mos_inputs('MNCM31')].values
                              , vss.reshape([-1,1]) ]
                            , axis = 1)
    ls1_in  = np.concatenate( [ i7.reshape([-1,1])
                              , df[mos_inputs('MNLS11')].values
                              , np.negative(vy.reshape([-1,1])) ]
                            , axis = 1)
    ls2_in  = np.concatenate( [ i7.reshape([-1,1])
                              , df[mos_inputs('MPLS21')].values
                              , (vdd - vw).reshape([-1,1]) ]
                            , axis = 1)
    dp1_in  = np.concatenate( [ i12.reshape([-1,1])
                              , df[mos_inputs('MNDP11')].values
                              , vcm1.reshape([-1,1]) ]
                            , axis = 1)
    dp2_in  = np.concatenate( [ i22.reshape([-1,1])
                              , df[mos_inputs('MPDP21')].values
                              , vdd - vcm2.reshape([-1,1]) ]
                            , axis = 1)
    rf1_in  = np.concatenate( [ i4.reshape([-1,1])
                              , gmid_rf.reshape([-1,1])
                              , fug_rf.reshape([-1,1])
                              , vds_rf.reshape([-1,1])
                              , vss.reshape([-1,1]) ]
                            , axis = 1)
    rf2_in  = np.concatenate( [ i5.reshape([-1,1])
                              , gmid_rf.reshape([-1,1])
                              , fug_rf.reshape([-1,1])
                              , np.negative(vds_rf.reshape([-1,1]))
                              , vss.reshape([-1,1]) ]
                            , axis = 1)
    cm1     = mos_predict(nmos, cm1_in)
    cm2     = mos_predict(pmos, cm2_in)
    cm3     = mos_predict(nmos, cm3_in)
    dp1     = mos_predict(nmos, dp1_in)
    dp2     = mos_predict(pmos, dp2_in)
    rf1     = mos_predict(nmos, rf1_in)
    rf2     = mos_predict(pmos, rf2_in)
    ls1     = mos_predict(nmos, ls1_in)
    ls2     = mos_predict(nmos, ls2_in)
    Wcm1    = cm1[:,0] / Mcm11
    Wcm2    = cm2[:,0] / Mcm21
    Wcm3    = cm3[:,0] / Mcm31
    Wdp1    = dp1[:,0] / Mdp11
    Wdp2    = dp2[:,0] / Mdp21
    Wls1    = ls1[:,0] / Mls11
    Wls2    = ls2[:,0] / Mls21
    Wrf1    = np.ones(num) * 0.25e-6
    Wrf2    = np.ones(num) * 0.25e-6
    Lcm1    = cm1[:,1]
    Lcm2    = cm2[:,1]
    Lcm3    = cm3[:,1]
    Ldp1    = dp1[:,1]
    Ldp2    = dp2[:,1]
    Lls1    = ls1[:,1]
    Lls2    = ls2[:,1]
    Lrf1    = (rf1[:,1] * Wrf1) / rf1[:,0]
    Lrf2    = (rf2[:,1] * Wrf2) / rf2[:,0]
    sizing  = np.concatenate([ s.reshape([-1,1]) for s in
                               [ Ldp1, Ldp2, Lcm1, Lcm2, Lcm3, Lls1, Lrf1, Lls2, Lrf2
                               , Wdp1, Wdp2, Wcm1, Wcm2, Wcm3, Wls1, Wrf1, Wls2, Wrf2
                               , Mdp11, Mdp12, Mdp21, Mdp22, Mcm11, Mcm12, Mcm13
                               , Mcm14, Mcm21, Mcm22, Mcm23, Mcm24, Mcm25, Mcm31
                               , Mcm32, Mls11, Mls12, Mrf11, Mls21, Mls22, Mrf21 ] ]
                            , axis = 1 )
    return pd.DataFrame(sizing, columns = geom_params('rfa'))

@case_distinction
def transform(ckt: Match(_), *args):
    raise ValueError(f'transform not defined for {ckt}')
