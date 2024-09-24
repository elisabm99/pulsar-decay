import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.special import erf
from astropy.cosmology import Planck18
from scipy.special import spence
from scipy.interpolate import interp1d
from scipy import integrate
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
from pytictoc import TicToc
import matplotlib.colors as colorss

np.seterr(all='warn')
# warnings.filterwarnings('error')


plt.rc('text', usetex = True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

timer = TicToc ()

pi = np.pi
arcmin = 1. / 60 * pi / 180 # arcmin to radians
degree = pi / 180 # degree to radians
cm = 3.24078e-22  # kpc
c = 3e5 # km/s
gram = 5.62e23 # GeV
G = 1.25865e-9 # (GeV / cm^3)^-1 (kpc)^-2
year = 3.05404e-4 # kpc

Msun = 1.9884e+30 * 10 ** 3  # grams
cosmo = Planck18.clone(H0=69)
rsun = 8.178 # 8.122  # kpc   Gravity collaboration 2019,

va = np.array([0, 245.6 / c, 0]) #np.array([0, 250 / c, 0]) #
dv=1e-3
tau0_ideal = 1e3 * year  # kpc
tau0_pulsar = 1.97e4 * year # kpc
radius_source = 3.24078e-16  # 3.24078e-16 # kpc. Corresponds to 10 km
fdeltanu = 0.84

idealized_sources_dic = {
    0: {
        'name_source': 'A',
        'l_source': 90. * degree,
        'b_source': 0. * degree,
        'd_source': 0.1,  # kpc
        'age_source': 5e3 * year,  # kpc
        's400': 1,  # Jy
        'age_over_tau0': 1.,
        'vs': 200 / c * np.array([0, 1, 0])  # velocity of the source in the X1, Y1, Z1 system
    },
    1: {
        'name_source': 'B',
        'l_source': 270. * degree,
        'b_source': 45. * degree,
        'd_source': 1.,  # kpc
        'age_source': 1e4 * year,  # kpc
        's400': 1,  # Jy
        'age_over_tau0': 2.,
        'vs': 200 / c / np.sqrt(2) * np.array([0, -1, 1])  # velocity of the source in the X1, Y1, Z1 system
    },
    2: {
        'name_source': 'C',
        'l_source': 0. * degree,
        'b_source': 90. * degree,
        'd_source': 0.1,  # kpc
        'age_source': 5e4 * year,  # kpc
        's400': 1,  # Jy
        'age_over_tau0': 3.,
        'vs': 200 / c * np.array([0, 1, 0])  # velocity of the source in the X1, Y1, Z1 system
    },
    3: {
        'name_source': 'D',
        'l_source': 0. * degree,
        'b_source': 0.5 * degree,
        'd_source': 3.,  # kpc
        'age_source': 1e5 * year,  # kpc
        's400': 1,  # Jy
        'age_over_tau0': 4.,
        'vs': 200 / c / np.sqrt(2) * np.array([-1, 0, 1])  # velocity of the source in the X1, Y1, Z1 system
    }
}

pulsar_dic = {
    0: {
        'name_source': 'J0835-4510',
        'l_source': 263.55183143 * degree, # 263.552 * degree,
        'b_source': -2.78731237 * degree , #-2.787 * degree,
        'd_source': 0.280,  # kpc
        'age_source': 1.13e4 * year,  # kpc
        's400': 5,  # Jy
        'pmra': -49.68,  # mas/yr,
        'pmdec': 29.9,  # mas/yr,
        'Pdot': 1.25008E-13,
        'radec': SkyCoord(ra='08h35m20.61149', dec='-45d10m34.8751', distance=0.280*u.kpc),
        'vs': np.array([59.43964813199587, -7.380416385615263, -13.523109398628108]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'taus' : 0.03955E-3, # s
        'P' : 0.089328385024,
        'W' : 1.7e-3,
        'vl' :-2.1801302456993802e-07 / year, # rad / kpc
        'vb': -4.922511312505952e-08  / year # rad / kpc
},
    1: {
        'name_source': 'J0332+5434',
        'l_source': 144.995 * degree,
        'b_source': -1.221 * degree,
        'd_source': 1.695,  # kpc
        'age_source': 5.53e6 * year,  # kpc
        's400': 1.5,  # Jy
        'pmra': 16.97,  # mas/yr,
        'pmdec': -10.37,  # mas/yr,
        'radec': SkyCoord(ra='03h32m59.4096', dec='54d34m43.329', distance=1.695*u.kpc),
        'vs': np.array([64.50745446511519, 92.94905588382463, -22.579315531491062]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'taus': 2.55E-8, # s
        'P': 0.714519699726,
        'W': 6.6e-3,
        'vl' : 6.796675420320407e-08 / year, # rad / kpc
        'vb': -1.3564185282710303e-08 / year # rad / kpc
    
    },
    2: {
        'name_source': 'J1644-4559',
        'l_source': 339.193 * degree,
        'b_source': -0.195 * degree,
        'd_source': 4.5,  # kpc
        'age_source': 3.59e5 * year,  # kpc
        's400': 0.375,  # Jy
        'pmra': 0,  # mas/yr,
        'pmdec': 0,  # mas/yr,
        'radec': SkyCoord(ra='16h44m49.273', dec='-45d59m09.71', distance=4.5*u.kpc),
        'vs': np.array([-0.0, -0.0, 0.0]) / c,  # velocity of the source in the X1, Y1, Z1 system
        'taus': 1.016E-02 , # s
        'P': 0.4550782016717,
        'W': 8.0e-3,
        'vl' :0 , # rad / kpc
        'vb': 0  # rad / kpc
    },
    3: {
        'name_source': 'J1645-0317',
        'l_source': 14.11419219 * degree,
        'b_source': 26.06162766 * degree,
        'd_source': 3.846,  # kpc
        'age_source': 3.45e6 * year,  # kpc
        's400': 0.393,  # Jy
        'pmra': -1.035,  # mas/yr,
        'pmdec': 20.5,  # mas/yr,
        'radec': SkyCoord(ra='16h45m02.0406', dec='-03d17m57.819', distance=3.846*u.kpc),
        'vs': np.array([167.44066802887403, -274.12225655648825, 195.35191608871827]) / c,  # velocity of the source in the X1, Y1, Z1 system
        'taus': 9.4E-8 , # s
        'P': 0.387689698034,
        'W': 3.8e-3,
        'vl' : 9.036406114887555e-08 / year, # rad / kpc
        'vb': 5.756128071890337e-08 / year # rad / kpc # rad / year
    },
    4: {
        'name_source': 'J1752-2806',
        'l_source': 1.540 * degree,
        'b_source': -0.961 * degree,
        'd_source': 0.2,  # kpc
        'age_source': 1.1e6 * year,  # kpc
        's400': 1.1,  # Jy
        'pmra': -4,  # mas/yr,
        'pmdec': -5,  # mas/yr,
        'radec': SkyCoord(ra='17h52m58.6896', dec='-28d06m37.3', distance=0.2*u.kpc),
        'vs': np.array([-0.16405684334941384, 5.806932026551248, 0.4747139102885267]) / c,
        # velocity of the source in the X1, Y1, Z1 system, # s
        'P': 0.56255763553,
        'W': 6.6e-3
    },
    5: {
        'name_source': 'J0437-4715',
        'l_source': 253.394 * degree,
        'b_source': -41.963 * degree,
        'd_source': 0.157,  # kpc
        'age_source': 6.64e9 * year,  # kpc
        's400': 0.550,  # Jy
        'pmra': 121.4385,  # mas/yr,
        'pmdec': -71.4754,  # mas/yr,
        'radec': SkyCoord(ra='04h37m15.8961737', dec='-47d15m09.110714', distance=0.157*u.kpc),
        'vs': np.array([-34.53040677620645, 55.786296408844024, 48.47564026337636]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'P': 0.005757451936712637,
        'W': 0.1410e-3
    },
    6: {
        'name_source': 'J0534+2200',
        'l_source': 184.558 * degree,
        'b_source': -5.784 * degree,
        'd_source': 2.0,  # kpc
        'age_source': 1.26e3 * year,  # kpc
        's400': 0.55,  # Jy
        'pmra': -14.7,  # mas/yr,
        'pmdec': 2.0,  # mas/yr,
        'radec': SkyCoord(ra='05h34m31.973', dec='22d00m52.06', distance=2.0*u.kpc),
        'vs': np.array([-3.1903991557697964, -86.10267158978502, -98.93563357716317]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'P': 0.0333924123  ,
        'W': 3.0e-3
    },
    7: {
        'name_source': 'J0953+0755',
        'l_source': 228.908 * degree,
        'b_source': 43.697 * degree,
        'd_source': 0.261,  # kpc
        'age_source': 1.75e7 * year,  # kpc
        's400': 0.4,  # Jy
        'pmra': -2.09,  # mas/yr,
        'pmdec': 29.46,  # mas/yr,
        'radec': SkyCoord(ra='09h53m09.3097', dec='07d55m35.75', distance=0.261*u.kpc),
        'vs': np.array([17.451165772916166, -30.08960433959107, 11.728991226718222]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'P': 0.2530651649482,
        'W': 8.6e-3
    },
    'snr': {
        'name_source': 'W50',
        'l_source': 35.39226144 * degree,
        'b_source': -4.61033397 * degree,
        'd_source': 5,  # kpc
        'age_source': 6e4 * year,  # kpc
        's400': 161.427 ,  # Jy
        'pmra': 0,  # mas/yr,
        'pmdec': 0,  # mas/yr,
        'radec': SkyCoord(ra='19h12m20s', dec='+0d04m55s',
                           distance=5 * u.kpc),
        'vs': np.array([0,0,0]) / c,
        # velocity of the source in the X1, Y1, Z1 system
        'P': 0.,
        'W': 0.
    }

}


nus_SKA1low = np.array([60, 96, 132, 183, 253, 350]) # MHz
sigma_SKA1low = np.array([11050, 3261, 1841, 1258, 973, 794]) # nuJy / beam. 1hr integration. \Delta\nu/\nu = 10^-4
beam_SKA1low_min = np.array([23.5, 17.0, 12.3, 8.9, 6.4, 4.6])  # arcsec
beam_SKA1low     = np.array([1175, 850, 614, 444, 321, 232])  # arcsec

nus_SKA1mid = np.array([0.48, 0.65, 0.89, 1.21, 1.65, 2.25, 3.07, 4.18, 5.70, 7.78, 10.61, 14.46, 19.72, 26.89, 36.67, 50]) * 1000 # MHz
sigma_SKA1mid = np.array([1176,    560,  303,   186,    137,  113,    99,   109,    95,    89,    85,    85,    91,   116,   121,   209]) # nuJy / beam. 1hr integration. \Delta\nu/\nu = 10^-4
beam_SKA1mid_min = np.array([2.031, 1.489, 1.092, 0.801, 0.587, 0.431, 0.316, 0.232, 0.170, 0.125, 0.091, 0.067, 0.049, 0.036, 0.026, 0.019])  # arcsec
beam_SKA1mid =     np.array([270.8, 198.6, 145.6, 106.8,  78.3,  57.4,  42.1,  30.9,  22.7,  16.6,  12.2,   8.9,   6.6,  4.8,    3.5,   2.6])  # arcsec
f_sigma_SKA1 = interp1d(np.concatenate((nus_SKA1low, nus_SKA1mid)), np.concatenate((sigma_SKA1low, sigma_SKA1mid)))
f_beam_SKA1 = interp1d(np.concatenate((nus_SKA1low, nus_SKA1mid)), np.concatenate((beam_SKA1low, beam_SKA1mid)))

nus_LOFAR = np.array([30, 45, 60, 75, 120, 150, 180, 200, 210, 240]) # MHz
sigma_LOFAR = np.array([3.8, 3.1, 2.6, 4.5, 0.20, 0.16, 0.20, 0.32, 0.32, 0.73]) * 1e3 # nuJy / beam. 8hr integration, 3.66MHz BW
beam_LOFAR = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3]) * 100 # arcsec
f_sigma_LOFAR = interp1d(nus_LOFAR, sigma_LOFAR)
f_beam_LOFAR = interp1d(nus_LOFAR, beam_LOFAR)

#################################################
#################################################
#################################################

def thetad_behind_earth(thetai, t):

    return np.arcsin( np.sin(thetai) / np.sqrt( t**2 + 1 + 2 * t * np.cos(thetai)) )

def thetad_behind_source(thetai, t):
    
    return np.arcsin(-np.sin(thetai) / np.sqrt(t ** 2 + 1 - 2 * t * np.cos(thetai)))

def exponential_factor(thetad, dv):
    
    return np.exp(-0.125 * (thetad / dv)**2 )

def flux(theta, alpha, g11, dv):
    # Ratio between echo flux and source flux. g11 = g 10^-11/GeV. For other parameters, see latex note.
    return 4.21251e-8 * g11 * (
                2 * np.sqrt(2 / np.pi) * dv * alpha * (np.exp(-theta ** 2 / (8 * dv ** 2 * alpha ** 2)) - 1
                                                       ) + theta * erf(theta / (2 * np.sqrt(2) * dv * alpha)))

def get_NFW_parameters():
    
    # print(Planck18)
    # print(cosmo)
    
    # nfw = NFW(mass=6.5e11, concentration=14.5, redshift=0.0, massfactor=('virial', 183 * u.kpc), cosmo=cosmo) # DR3+ arxiv:2302.01379
    # nfw = NFW(mass=4.8e11, concentration=19.6, redshift=0.0, massfactor=('virial', 166 * u.kpc), cosmo=cosmo) # DR3 arxiv:2302.01379
    # nfw = NFW(mass=8e11, concentration=13, redshift=0.0, massfactor=('virial', 197 * u.kpc), cosmo=cosmo) # E19 arxiv:2302.01379
    #
    # rho0 = nfw.rho_scale.value * 3.80181e-8 # GeV cm^-3
    # r0 = nfw.r_s.value * Msun ** (1 / 3) * cm # kpc
    
    # Francesca's values
    r0 = 20 # kpc
    rho0 = 10.69  * 1e-3 * Msun * gram * (1e3 * cm)**3 #* rsun/r0 * (1 + rsun/r0)**2


    return rho0, r0

def rhoNFW(r):
    # rho0 = 0.51 #± 0.09 GeV/cm3
    # r0 =  8.1 # ±0.7 kpc
    
    return rho0 / ( r/r0 * (1 + r/r0)**2 )

def get_velocity_components_in_source_system(va, ls, bs, phii, isOpposite):
    # va is the DM velocity. Must be an array with 3 elements: the components of the velocity in the X1, Y1, Z1 system
    # ls, bs are the galactic coordinate of the source
    # phii is the azimuthal angle of the los wrt the Z1 axis in the plane perpendicular to the line of sight towards the source
    
    a = 1 if isOpposite else -1
    
    nx = np.array([
        np.cos(ls) * np.cos(phii) * np.sin(bs) + a * np.sin(ls) * np.sin(phii),
        np.cos(phii) * np.sin(bs) * np.sin(ls) - a * np.cos(ls) * np.sin(phii),
        np.cos(bs) * np.cos(phii)]).T
    
    ny = np.array([
        a * np.cos(phii) * np.sin(ls) - np.cos(ls) * np.sin(bs) * np.sin(phii),
        -a * np.cos(ls) * np.cos(phii) - np.sin(bs) * np.sin(ls) * np.sin(phii),
        -np.cos(bs) * np.sin(phii)]).T
    
    nz = np.array([
        a * np.cos(bs) * np.cos(ls),
        a * np.cos(bs) * np.sin(ls),
        -a * np.sin(bs)]).T

    return np.dot(nx,va), np.dot(ny,va), np.dot(nz,va)

def get_vpar_vperp_vy(va, ls, bs, thetai, phii, isOpposite):
    # vx, vy, vz are the components of the DM velocity in the system defined in the xyz centered at the source
    vx, vy, vz = get_velocity_components_in_source_system(va, ls, bs, phii, isOpposite)

    vperp = -vx * np.cos(thetai) + vz * np.sin(thetai)
    vpar = -vx * np.sin(thetai) - vz * np.cos(thetai)
    return vpar,  vperp, vy
    
def get_dv(r):
    if not isinstance(r, list) and not isinstance(r, np.ndarray):
        r = np.array([r])
        
    x = r / r0
    y = 1 + x
    tmp = pi ** 2 - np.log(x) +  3 * np.log(y) ** 2 - 1 / x - 1 / y**2 - 6 / y + np.log(y) * (
            1 + 1 / x**2 - 4 / x - 2 / y) + 6 * spence(y)
    idx = np.where(np.abs((tmp) < 1e-12))
    tmp[idx] = 0
    
    try:
        res = np.sqrt(4 * pi * G * rho0 * r0**2 * 0.5 * x * y**2 * tmp ) # arXiv:0002395
    except RuntimeWarning:
        print('spence(y) =', spence(y))
        print()
        print((
            pi ** 2 - np.log(x) +  3 * np.log(y) ** 2 - 1 / x - 1 / y**2 - 6 / y + np.log(y) * (
            1 + 1 / x**2 - 4 / x - 2 / y) + 6 * spence(y)) )
        sys.exit()
        
    if np.any(np.isnan(res)):
        idx = np.where(np.isnan(res))
        print('x =', x[idx])
        print('spence(y) =', spence(y[idx]))
        print()
        print(tmp)
    return res

def get_velocity_from_pm(dict):
    
    deg = u.degree

    radec_source = dict['radec']
    dist = dict['d_source']
    # print('radec_source =', radec_source)
    lb_source = radec_source.galactic
    
    pmra = dict['pmra'] * 2.77778e-7 # deg / yr
    pmdec = dict['pmdec'] * 2.77778e-7 # deg / yr
    radec_v = SkyCoord(ra=(radec_source.ra.degree + pmra)*deg, dec=(radec_source.dec.degree + pmdec)*deg, distance=dist*u.kpc)
    
    lb_v = radec_v.galactic

    # print('lb_source galactic =', lb_source)
    # print('lb_v galactic =', lb_v)
    # print()
    lb_source.representation_type = 'cartesian' # kpc /yr
    lb_v.representation_type = 'cartesian'# kpc /yr
    # print('lb_source cartesian =', lb_source)
    # print(lb_v)

    vX1 = -(lb_v.u - lb_source.u).value * 9.82304e8 # km/s
    vY1 = -(lb_v.v - lb_source.v).value * 9.82304e8 # km/s
    vZ1 = (lb_v.w - lb_source.w).value * 9.82304e8 # km/s
    
    print(vX1, vY1, vZ1)
    
    return np.array([vX1, vY1, vZ1])

def get_velocity_from_pm_galactic_coords(dict):
    deg = u.degree
    
    radec_source = dict['radec']
    dist = dict['d_source']
    lb_source = radec_source.galactic
    pmra = dict['pmra'] * 2.77778e-7  # deg / yr
    pmdec = dict['pmdec'] * 2.77778e-7  # deg / yr
    radec_v = SkyCoord(ra=(radec_source.ra.degree + pmra) * deg, dec=(radec_source.dec.degree + pmdec) * deg,
                       distance=dist * u.kpc)
    
    lb_v = radec_v.galactic
    
    # print('lb_source galactic =', lb_source)
    # print('lb_v galactic =', lb_v)
    # print((lb_v.l - lb_source.l).degree)
    # print((lb_v.b - lb_source.b).degree)
    
    return (lb_v.l - lb_source.l).rad, (lb_v.b - lb_source.b).rad # deg / yr

def get_pulsar_flux(i_pulsar):
    if i_pulsar == 0:
        name = 'J0835-4510'
        model = 'broken_power_law'
        vb = 786.9
        a1 = -0.51096
        a2 = -2.02643
        c = 1.23381
        v0 = 5109.4
        freqs = np.sort(np.array(
            [154.0, 8356, 950.0, 952.5, 17000.0, 24000.0, 312, 289, 344, 305, 320, 328, 336, 352, 728, 1382, 3100,
             408.0, 4820.0, 1360, 151.5, 76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197,
             204.5, 212, 220, 227, 154.24, 97500, 145000, 233000, 343500, 147.5, 400]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = np.empty_like(v)
        S[np.where(v <= vb)] = c * (v[np.where(v <= vb)] / v0) ** a1
        S[np.where(v > vb)] = c * (v[np.where(v > vb)] / v0) ** a2 * (vb / v0) ** (a1 - a2)
    
    if i_pulsar == 1:
        name = 'J0332+5434'
        model = 'low_frequency_turn_over_power_law'
        vpeak = 161.5
        a = -2.42975
        c = 1.66104
        beta = 0.69976
        v0 = 753.3
        freqs = np.sort(np.array(
            [14800, 22700, 39.0, 61.0, 85.0, 102.5, 65.0, 4820.0, 102.5, 350.0, 408, 606, 925, 1408, 25, 147.5, 35.1,
             45.0, 49.8, 55.0, 65.0, 74.0]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = c * (v / v0) ** a * np.exp(a / beta * (v / vpeak) ** -beta)
    
    if i_pulsar == 2:
        name = 'J1644-4559'
        model = 'broken_power_law'
        vb = 729.9
        a1 = 2.50422
        a2 = -2.07414
        c = 7.98889
        v0 = 1583.5
        freqs = np.sort(np.array(
            [14800, 22700, 39.0, 61.0, 85.0, 102.5, 65.0, 4820.0, 102.5, 350.0, 408, 606, 925, 1408, 25, 147.5, 35.1,
             45.0, 49.8, 55.0, 65.0, 74.0]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = np.empty_like(v)
        S[np.where(v <= vb)] = c * (v[np.where(v <= vb)] / v0) ** a1
        S[np.where(v > vb)] = c * (v[np.where(v > vb)] / v0) ** a2 * (vb / v0) ** (a1 - a2)
    
    if i_pulsar == 3:
        name= 'J1645-0317'
        model = 'low_frequency_turn_over_power_law'
        vpeak = 115.8
        a = -5.23562
        c = 201375.80129
        beta = 0.21259
        v0 = 1063.2
        freqs = np.sort(np.array(
            [22700, 61.0, 85.0, 102.5, 950.0, 185.0, 65.0, 180, 117, 336, 86, 102, 133, 148, 159, 195, 200, 211, 289,
             305, 320, 328, 344, 352, 408.0, 4820.0, 102.5, 843, 350.0, 1360, 151.5, 1369, 408, 606, 1408, 135.25,
             147.5, 49.8, 64.5, 79.2]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = c * (v / v0) ** a * np.exp(a / beta * (v / vpeak) ** -beta)
    
    if i_pulsar == 4:
        name = 'J1752-2806'
        model = 'low_frequency_turn_over_power_law'
        vpeak =    141.7
        a = -8.00000
        c = 2572777.04479
        beta = 0.26026
        v0 =   1176.7
        freqs = np.sort(np.array(
            [22700, 154.0, 8356, 61.0, 85.0, 102.5, 950.0, 800.0, 185.0, 728, 3100, 408.0, 4820.0, 1577.0, 843, 350.0, 1360, 151.5, 408, 606, 1408, 147.5, 1400, 78.2]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = c * (v / v0) ** a * np.exp(a / beta * (v / vpeak) ** -beta)
    
    if i_pulsar == 5:
        name = 'J0437-4715'
        model = 'broken_power_law'
        vb = 2346.8
        a1 = -0.92813
        a2 = -2.48830
        c = 0.13715
        v0 = 1136.7
        freqs = np.sort(np.array(
            [430, 1520, 2360, 1284, 154.0, 185.0, 17000.0, 150, 728, 1382, 3100, 1400.0, 436, 660, 1400, 1660, 436.0, 843, 151.5, 76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197, 204.5, 212, 220, 227, 154.24, 147.5, 730, 1400, 3100]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = np.empty_like(v)
        S[np.where(v <= vb)] = c * (v[np.where(v <= vb)] / v0) ** a1
        S[np.where(v > vb)] = c * (v[np.where(v > vb)] / v0) ** a2 * (vb / v0) ** (a1 - a2)
    
    if i_pulsar == 6:
        name = 'J0534+2200'
        model = 'broken_power_law'
        vb = 416.8
        a1 = 2.96462
        a2 = -3.22458
        c = 0.50760
        v0 = 379.9
        freqs = np.sort(np.array(
            [102.5, 149, 408, 606, 925, 1408, 154.24, 147.5, 400]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = np.empty_like(v)
        S[np.where(v <= vb)] = c * (v[np.where(v <= vb)] / v0) ** a1
        S[np.where(v > vb)] = c * (v[np.where(v > vb)] / v0) ** a2 * (vb / v0) ** (a1 - a2)
    
    if i_pulsar == 7:
        name = 'J0953+0755'
        model = 'low_frequency_turn_over_power_law'
        vpeak =    210.2
        a = -2.25616
        c = 0.40842
        beta = 2.10000
        v0 =    673.8
        freqs = np.sort(np.array(
            [22700, 154.0, 8356, 39.0, 61.0, 88.0, 102.5, 950.0, 185.0, 62.0, 70, 305, 336, 86, 102, 133, 148, 150, 159, 180, 195, 211, 227, 289, 328, 728, 1382, 3100, 408.0, 4820.0, 102.5, 843, 1360, 151.5, 408, 606, 925, 1408, 135.25, 154.24, 20, 25, 147.5]))
        v = np.linspace(np.amin(freqs), np.amax(freqs), num=2000)
        S = c * (v / v0) ** a * np.exp(a / beta * (v / vpeak) ** -beta)
        
    return interp1d(v, S) #, bounds_error=False, fill_value='extrapolate')

def get_pulsar_beam_opening(i_pulsar):
    return 5.4 * pi /180 * (pulsar_dic[i_pulsar]['P'])**-0.5 # radians

def get_broadening_angle(nu, i_pulsar):  # nu must be in MHz
    taus = pulsar_dic[i_pulsar]['taus']
    dist = pulsar_dic[i_pulsar]['d_source']
    
    return np.sqrt(taus / dist * 9.72132e-12) * (1000 / nu) ** 2.2

def get_X1_over_xs(thetai, phii, dict, isOpposite):
    a = 1 if isOpposite else -1
    
    # Get l, b of the los
    ls = dict['l_source']
    bs = dict['b_source']
    
    uu = np.abs(thetai) * np.cos(phii)  # change in galactic latidute
    vv = np.abs(thetai) * np.sin(phii)  # negative change in galactic longitude
    
    if isOpposite:
        l_los = (ls + pi - vv) % (2 * pi)
    else:
        l_los = ls - vv
    
    b_los = -a * bs + uu
    
    # Get cartesian coordinates of a point along the los at distance xs from the sun in the X',Y',Z' coordinate system
    return -np.cos(b_los) * np.cos(l_los), l_los, b_los

def get_smax(thetai, vperp, omega, delta, isOpposite):
    if not isOpposite: # for behind echo use the fact that integrand is suppressed close to the source for thetai != 0. This helps avoid numerical problems.
        if 0 < np.abs(thetai) :
            k = 5
            smax = (-vperp + k  * dvref) / (thetai * omega)
        else:
            smax = 1 / delta
    else:
        smax = 1.
        
    if isOpposite:
        smax = 1 - 1e-4
        
    return smax

def get_smin(thetai, age_source, xs, isOpposite, isSourceVelocity, vss):
    a = 1 if isOpposite else -1

    if not isSourceVelocity:
        tmax = 0.5 * (age_source ** 2 - xs ** 2) / xs / (age_source + a * xs * np.cos(thetai))
    else:
        vs_x, vs_y, vs_z = vss
        vs = np.sqrt(vs_x ** 2 + vs_y ** 2 + vs_z ** 2)
        tmax = 1. / xs * (
                    (age_source + xs) * (- age_source + age_source * vs ** 2 + xs + vs ** 2 * xs - 2 * a * vs_z * xs)
                    ) / (2. * (
                    - age_source + age_source * vs_z * np.cos(thetai) - a * xs * np.cos(thetai) + vs_z * xs * np.cos(
                thetai) + age_source * vs_x * np.sin(thetai) + vs_x * xs * np.sin(thetai)))

    if (tmax <= 0 and isOpposite) or (tmax < np.cos(thetai) and not isOpposite):
        if thetai == 0:
            # print('tmax =', tmax)
            print('exiting', thetai, age_source, xs, tmax)
        return 0
    
    smin = 1. / np.sqrt(1 + tmax ** 2 + a * 2 * tmax * np.cos(thetai))

    return smin

def get_time_dependent_quantities(thetai, phii, ls, bs, xs, smin, smax, dict, isOpposite):
    a = 1 if isOpposite else -1
    try:
        tmax = -a * np.cos(thetai) + np.sqrt(1 - (smin * np.sin(thetai)) ** 2) / smin
        tmin = -a * np.cos(thetai) + np.sqrt(1 - (smax * np.sin(thetai)) ** 2) / smax
    

        ts = np.logspace(np.log10(tmin), np.log10(tmax), num=20000)
    except Warning:
        print('smin, smax =', smin, smax)
        print('tmin, tmax =', tmin, tmax)
        print('thetai, a =', thetai, a)
        print(-a * np.cos(thetai))
        print(np.sqrt(1 - (smax * np.sin(thetai)) ** 2), smax )

    xd = xs * ts
    
    vs_x, vs_y, vs_z = np.array(get_velocity_components_in_source_system(dict['vs'], ls, bs, phii, isOpposite))
    # vs_x, vs_y, vs_z = [0,0,0]
    # vs_y = 0
    
    vs = np.sqrt(vs_x ** 2 + vs_y ** 2 + vs_z ** 2)
    
    b = xs * (vs ** 2 - a * vs_z) - xd * (vs_z * np.cos(thetai) + vs_x * np.sin(thetai))
    
    cc = xd ** 2 + xs ** 2 * (1 - 2 * a * vs_z + vs ** 2) + 2 * xd * xs * (
            (a - vs_z) * np.cos(thetai) - vs_x * np.sin(thetai))
    
    Delta = b ** 2 + cc * (1 - vs ** 2) - 2 * b * xd + vs ** 2 * xd ** 2
    idx = np.where(Delta < 0)[0]
    if len(idx) != 0:
        if Delta[idx] > -3e-11:
            Delta[idx] = 0
    
    time = 1 / (1 - vs ** 2) * (-xd + b - np.sqrt(Delta))  # kpc
    
    time_plus_xs = time + xs
    xs_time = np.sqrt((vs_x * time_plus_xs) ** 2 + (vs_y * time_plus_xs) ** 2 + (a * xs - vs_z * time_plus_xs) ** 2)
    xds_time = np.sqrt(
        (xd * np.sin(thetai) - vs_x * time_plus_xs) ** 2 + (vs_y * time_plus_xs) ** 2 + (
                    xd * np.cos(thetai) + a * xs - vs_z * time_plus_xs) ** 2
    )
 
    
    xds_time[np.where(xds_time == 0)] = (1 - tmin) ** 2
    xs_tobs = np.sqrt((vs_x * xs) ** 2 + (vs_y * xs) ** 2 + (a * xs - vs_z * xs) ** 2)


    Deltax = xd * np.sin(thetai) - vs_x * time_plus_xs
    Deltay = - vs_y * time_plus_xs
    Deltaz = xd * np.cos(thetai) + a * xs - vs_z * time_plus_xs

    thetas = np.arctan2(np.sqrt(Deltax ** 2 + Deltay ** 2), Deltaz)
    phis = np.arctan2(Deltay, Deltax)
    
    return xs_time, xds_time, xs_tobs, ts, time, thetas, phis

def integrate_los(thetai, phii, dict, omega, i_plot, isOpposite, isIdealizedSources):
    # Calculates \int_{r_min}^R dx_d x_s/x_{ xs }^2 \rho(r)/\rho(rs) exp[- \theta_i^2 x_s^2 / (8 \delta v^2 x_{ xs }^2)]
    # \rho(rs) is the DM energy density at the location of the source
    # omega in units of the axion mass
    
    a = 1 if isOpposite else -1
    
    epsilon = 2 * omega - 1
    
    panels_with_xs = [1, 2, 5, 6, 8]  # in these panels, we consider the actual distance of the source, not the default value
    panels_with_age = [4, 5, 6, 8]  # in these panels, we take into account the age of the source.
    panels_with_dm_velocity = [3, 8]  # in these panels, we take into account the age of the source.
    
    # Get l, b of the los
    ls = dict['l_source']
    bs = dict['b_source']
    if i_plot in panels_with_xs:
        xs = dict['d_source']
    else:
        xs = xsref
    
    # Get cartesian coordinates of a point along the los at distance xs from the sun in the X',Y',Z' coordinate system
    X1, l_los, b_los = get_X1_over_xs(thetai, phii, dict, isOpposite)
    X1 = xs * X1
    
    # Get DM velocity components
    if i_plot in panels_with_dm_velocity:
        vpar, vperp, vy = get_vpar_vperp_vy(va, ls, bs, thetai, phii, isOpposite)
        vx, vy, vz = get_velocity_components_in_source_system(va, ls, bs, phii, isOpposite)
    else:
        vpar, vperp, vy = [0, 0, 0]
    
    # Set up integration limits
    delta = radius_source / xs
    
    if i_plot in panels_with_age:
        age_source = dict['age_source']
        
        if not isIdealizedSources:
            age_source = age_source + xs
        
        vss = get_velocity_components_in_source_system(dict['vs'], ls, bs, phii, isOpposite)
        smin = get_smin(thetai, age_source, xs, isOpposite, isSourceVelocity=True, vss=vss)
        
    else:
        smin = 1e-10  # # It is equivalent to integrating in xd until 1/smin times the distance of the source
    
    smax = get_smax(thetai, vperp, omega, delta, isOpposite)
    
    # Set up integrand funtion
    ss = np.logspace(np.log10(smin), np.log10(smax), num=20000, endpoint=True)
    tmp = -a * np.cos(thetai) + np.sqrt(1 - (ss * np.sin(thetai)) ** 2) / ss
    rs = np.sqrt((xs * tmp) ** 2 + rsun ** 2 + 2 * rsun * X1 * tmp)
    
    try:
        sqrtfactor = np.sqrt(1 - ss ** 2 * np.sin(thetai) ** 2)
    except RuntimeWarning:
        print('RuntimeWarning!')
        print('thetai =', thetai)
        sys.exit()
    
    # 0 -> rho, 1 -> rho, xs
    if i_plot in [0, 1]:
        rhos = rhoNFW(rs) / rhoref  # rhoNFW(rs[0])
        f = rhos * np.exp(
            - 0.5 * (epsilon / dvref) ** 2
            - 0.5 * ((a * omega * thetai * ss) / dvref) ** 2
        ) / sqrtfactor
            
    # delta v, xs and rho
    elif i_plot == 2:
        
        rhos = rhoNFW(rs) / rhoref
        dvs = get_dv(rs)
        idx = np.where(dvs != 0)
        f = np.zeros_like(ss)
        
        f[idx] = rhos[idx] * (dvref / dvs[idx]) ** 3 * np.exp(
            - 0.5 * (epsilon / dvs[idx]) ** 2
            - 0.5 * ((a * omega * thetai * ss[idx]) / dvs[idx]) ** 2
        ) / sqrtfactor[idx]
    
    # v DM average
    elif i_plot == 3:
        
        f = np.exp(
            - 0.5 * (vy / dvref) ** 2
            - 0.5 * ((epsilon - vpar) / dvref) ** 2
            - 0.5 * ((a * omega * thetai * ss - vperp) / dvref) ** 2
        ) / sqrtfactor
    
    # 4: t_age, 5: age_source, xs
    elif i_plot == 4 or i_plot == 5:
        f = np.exp(
            - 0.5 * (epsilon / dvref) ** 2
            - 0.5 * ((a * omega * thetai * ss) / dvref) ** 2
        ) / sqrtfactor

    # t_age, xs, L(t)
    elif i_plot == 6:
        
        xd = xs * (-a * np.cos(thetai) + np.sqrt(1 - (ss * np.sin(thetai)) ** 2) / ss)
        xds = np.sqrt(xd ** 2 + xs ** 2 + 2 * a * xd * xs * np.cos(thetai))
        tau0 = tau0_ideal if isIdealizedSources else tau0_pulsar
        cE = 1 if isIdealizedSources else 0.3
        
        fluxfactor = ((1 + (age_source - xs) / tau0) / (1 + (age_source - xd - xds) / tau0)) ** cE
        
        f = fluxfactor * np.exp(
            - 0.5 * (epsilon / dvref) ** 2
            - 0.5 * ((a * omega * thetai * ss) / dvref) ** 2
        ) / sqrtfactor
    
    # 7: pulsar velocity
    elif i_plot == 7:
        # the ts returned here are different than the tmps used to calculate rs, but it doesn't matter cause here we're not considering \rho and dv(r)
        xs_time, xds_time, xs_tobs, ts, time, thetas_time, phis_time = get_time_dependent_quantities(thetai, phii, ls, bs, xs, smin, smax, dict, isOpposite)
        thetak = pi - thetai
        thetad = pi - thetak - thetas_time
        xx = omega * thetad * np.cos(phis_time) * np.cos(thetak) - (omega * (1-np.cos(phis_time)) + epsilon * np.cos(phis_time)) * np.sin(thetak)
        yy = (omega * thetad * np.cos(thetak) + (omega - epsilon) * np.sin(thetak)) * np.sin(phis_time)
        zz = epsilon * np.cos(thetak) + omega * thetad * np.sin(thetak)
        
        try:
            f = np.exp(
                - 0.5 * (xx**2 + yy**2 + zz**2)/ dvref ** 2
            ) * (xs_tobs / xds_time) ** 2
        except RuntimeWarning:
            print('thetai =', thetai)
        
        g = np.trapz(f, x=ts)
        
        return g
    
    try:
        g = np.trapz(f, x=ss)  # rhos
    except RuntimeWarning:
        print('RuntimeWarning!')
        print(smax, thetai, np.sqrt(1 + 0.5 * ss ** 2 * (np.cos(2 * thetai) - 1)))
    
    return g

def integrate_los_for_flux(thetai, phii, ipulsar, dict, isOpposite):  # all effects
    a = 1 if isOpposite else -1
    
    # Get l, b of the los
    ls = dict['l_source']
    bs = dict['b_source']
    xs = dict['d_source']
    
    # Get cartesian coordinates of a point along the los at distance xs from the sun in the X',Y',Z' coordinate system
    X1, l_los, b_los = get_X1_over_xs(thetai, phii, dict, isOpposite)
    X1 = xs * X1
    
    # Get DM velocity components
    vpar, vperp, vy = get_vpar_vperp_vy(va, ls, bs, thetai, phii, isOpposite)
    vx, vy, vz = get_velocity_components_in_source_system(va, ls, bs, phii, isOpposite)
    
    # Set up integration limits
    delta = radius_source / xs
    age_source = dict['age_source'] #+ xs
    
    # Get source velocity
    vss = get_velocity_components_in_source_system(dict['vs'], ls, bs, phii, isOpposite)
    np.set_printoptions(suppress=True)
    
    # Get integration limits
    smin = get_smin(thetai, age_source, xs, isOpposite, isSourceVelocity=False, vss=vss)
    smax = get_smax(thetai, vperp, 0.5, delta, isOpposite)
    
    xs_time, xds_time, xs_tobs, ts, time, thetas_time, phis_time = get_time_dependent_quantities(thetai, phii, ls, bs, xs, smin,
                       smax, dict, isOpposite)
    thetak = pi - thetai
    thetad = pi - thetak - thetas_time
    alpha = np.cos(thetak)**2 * np.cos(phis_time) + np.sin(thetak)**2
    beta = np.sin(2 * thetak) * np.sin(phis_time / 2)**2
    gamma = np.sin(thetas_time) * np.sin(phis_time)
    deltaa = np.cos(thetak)**2 + np.sin(thetak)**2 * np.cos(phis_time)
    eta = np.sin(thetak)**2 * (1 - np.cos(phis_time))
    
    a = gamma**2 + 4*deltaa**2 + 4*deltaa*eta + eta**2 +  2*beta*(alpha + 2*deltaa + eta)*thetad +alpha**2*thetad**2 + beta**2*(1 + thetad**2)
    b = -beta**2 + vy*gamma - gamma**2 - 2*vpar*deltaa -  2*deltaa**2 - vpar*eta - deltaa*eta - beta*(vpar +alpha + deltaa)*thetad - vperp*(beta +alpha*thetad)
    cc = vpar**2 + vperp**2 + vy**2 + 2*vperp*beta +beta**2 - 2*vy*gamma + gamma**2 +  2*vpar*deltaa + deltaa**2
    
    exponent = (b**2 - a * cc) / a

    xd = xs * ts
    cE = 0.3
    fluxfactor = ((1 + (age_source - xs_time) / tau0_pulsar) / (1 + (age_source - xd - xds_time) / tau0_pulsar)) ** cE
    rs = np.sqrt((xs * ts) ** 2 + rsun ** 2 + 2 * rsun * X1 * ts)
    rhos = rhoNFW(rs)
    dvs = get_dv(rs)
    idx = np.where(dvs != 0)
    f = np.zeros_like(ts)
    
    f[idx] = fluxfactor * rhos[idx] / dvs[idx] ** 2 * np.exp(
        0.5 * exponent / dvs[idx]**2
    ) * (xs_tobs / xds_time[idx]) ** 2
    

    # Handle finite size of pulsar beam
    idx_cut = np.where(thetas_time > get_pulsar_beam_opening(ipulsar))
    f[idx_cut] = 0
    
    return np.trapz(f[idx], x=ts[idx])

def integrate_los_collinear(dict):
    
    # Get l, b of the los
    ls = dict['l_source']
    bs = dict['b_source']
    
    xs = dict['d_source']
    
    # Get cartesian coordinates of a point along the los at distance xs from the sun in the X',Y',Z' coordinate system
    X1 = -xs * np.cos(bs) * np.cos(ls)
    
    tmax = 1.
    tmin = 0
    ts = np.linspace(tmin, tmax, num=5000)
    
    rs = np.sqrt((xs * ts) ** 2 + rsun ** 2 + 2 * rsun * X1 * ts)
    rhos = rhoNFW(rs)

    return np.trapz(rhos, x=ts)

#################################################
##                 Make forecast               ##
#################################################
def get_SKA1_beam(nu):
    # nu must be in MHz
    return f_beam_SKA1(nu)  # arcsec

def get_SKA1_sigma(nu, deltanu_over_nu, t):
    # nu must be in MHz
    # t must be in hours
    
    return f_sigma_SKA1(nu) * np.sqrt(1 / t) * np.sqrt(1e-4 / deltanu_over_nu) # muJy/ beam

def get_LOFAR_beam(nu):
    # nu must be in MHz
    return f_beam_LOFAR(nu)  # arcsec

def get_LOFAR_sigma(nu, deltanu_over_nu, t):
    # nu must be in MHz
    # t must be in hours
    deltanu = deltanu_over_nu * nu
    return f_sigma_LOFAR(nu) * np.sqrt(8 / t) * np.sqrt(3.66 / deltanu)  # muJy/ beam

def integrate_beam(thetai_c, phii_c,  ipulsar, dicts, beam, thetalim, isOpposite):
    
    X_c = thetai_c * np.cos(phii_c)
    Y_c = thetai_c * np.sin(phii_c)
    
    # timer.tic()
    nx = 50
    ny = 50

    dx = beam/nx
    
    if thetai_c == 0:
        xpos = np.logspace(-16, np.log10(beam-dx) , num=nx, endpoint=True)
        ypos = np.logspace(-16, np.log10(beam-dx) , num=ny, endpoint=True)

        xs = np.concatenate((-np.flip(xpos), xpos))
        ys = np.concatenate((-np.flip(ypos), ypos))

    else:
        xs = np.linspace(-beam+dx, beam-dx, num=(nx-2), endpoint=True)
        ys = np.linspace(-beam, beam, num=ny, endpoint=True)

    intensities = np.zeros_like(xs)
    for i, x in enumerate(xs):
        lim = np.sqrt(beam**2 - x**2)
        tmp = np.zeros_like(ys)

        for j, y  in enumerate(ys):

            if -lim < y < lim:
                X = X_c + x
                Y = Y_c + y
                thetai = np.sqrt(X**2 + Y**2)
                phii = np.arctan2(y,x)

                if thetai < thetalim:
                    tmp[j] = 0
                else:
                    tmp[j] =  integrate_los_for_flux(thetai, phii, ipulsar, dicts, isOpposite=isOpposite)
                
                if np.isnan(tmp[j]):
                    print(thetai, phii,tmp[j])
                

        intensities[i] = np.trapz(tmp, x=ys)
    # print(intensities)
    res = np.trapz(intensities, x=xs)
    return res
    # timer.toc()
        # print('intensities[i] =', intensities[i])

    # timer.tic()
    # print(integrate.dblquad(f, -beam, beam, lambda x: -np.sqrt(beam**2 - x**2), lambda x: np.sqrt(beam**2 - x**2),
    #       args=( X_c, Y_c, ipulsar, dicts, isOpposite), epsrel=5e-2        ))
    # # integrate.dblquad(f, -beam, beam, lambda x: -np.sqrt(beam**2 - x**2), lambda x: np.sqrt(beam**2 - x**2)) #, args=(1,))
    # timer.toc()
    
    # return integrate.dblquad(f, -beam, beam, lambda x: -np.sqrt(beam**2 - x**2), lambda x: np.sqrt(beam**2 - x**2),
    #       args=( X_c, Y_c, ipulsar, dicts, thetalim, isOpposite), epsrel=1e-1        )[0]

def calculate_flux_of_theta(nu, isOpposite): # DA CONTROLLARE only consistent for 400MHz
    thetai_obs = 20 * arcmin  # radians
    thetais_c = np.logspace(-14, np.log10(thetai_obs), num=200)  # center of the beam
    phiis_c = np.array([0, pi])
    # thetais = np.concatenate(([0], thetais))
    beam = get_SKA1_sigma(nu, deltanu_over_nu, 100) / 60 * arcmin  # radians
    
    rangelimit = 4
    myrange = range(rangelimit)
    # myrange = range(1)
    
 
    for ipulsar in myrange:
        dicts = pulsar_dic[ipulsar]
        d_source = dicts['d_source']
        s400 = dicts['s400']
        name_source = dicts['name_source']
        #
        thetalim = get_broadening_angle(nu, ipulsar)

        g = 10 ** -11  # GeV^-1
        deltanu = 2.8 * 400 * dvref  # MHs
        factor = fdeltanu * g ** 2 / 32 / deltanu * d_source * s400 / (
                    2 * pi) * 1.828e12  # conversion factor for GeV^-2/MHz kpc GeV/cm^3
        
        gs = np.empty([len(thetais_c), len(phiis_c)])
        
        print('\n*****************', name_source, isOpposite)
        for i, thetai_c in enumerate(thetais_c):
            for j, phii_c in enumerate(phiis_c):
                gs[i, j] = integrate_beam(thetai_c, phii_c, ipulsar, dicts, beam, thetalim, isOpposite=isOpposite)
        
        np.savez('gs_%s_%isOpposite=%s.npz' % (name_source, str(isOpposite)), gs=gs, thetais_c=thetais_c,
                 phiis_c=phiis_c)

def make_forecast(ipulsar, telescope):
    
    nsigma = 2
    dicts = pulsar_dic[ipulsar]
    d_source = dicts['d_source']
    name_source = dicts['name_source']

    print('\n*****************', name_source)
    
    if telescope=='SKA1':
        gs = np.empty([len(nus_SKA1low)+len(nus_SKA1mid)])
        bounds = np.empty([len(nus_SKA1low)+ len(nus_SKA1mid)])
        nus = np.concatenate((nus_SKA1low, nus_SKA1mid))
        beam_function = get_SKA1_beam
        sigma_function = get_SKA1_sigma
        
    if telescope=='LOFAR':
        gs = np.empty([len(nus_LOFAR)])
        bounds = np.empty([len(nus_LOFAR)])
        nus = nus_LOFAR
        beam_function = get_LOFAR_beam
        sigma_function = get_LOFAR_sigma
        
    for i, nu in enumerate(nus):

        beam = beam_function(nu) / 60 * arcmin  # radians. nu must be in MHz.
        sensitivity = sigma_function(nu, deltanu_over_nu, 100) * 1e-6  # Jy. nu must be in MHz. t must be in hours.
        thetalim = get_broadening_angle(nu, ipulsar)
        flux_source = get_pulsar_flux(ipulsar)
        
        try:
            flux = flux_source(nu)
        except ValueError:
            print('### ValueError', nu)
            bounds[i] = 1
            continue
            
        deltanu = 2.8 * dvref * nu  # MHs
        factor = fdeltanu / 16 / deltanu * d_source * flux / (8 * pi) * 1.828e12

        gs[i] = integrate_beam(0, 0, ipulsar, dicts, beam, thetalim, isOpposite=False)
        bounds[i] = np.sqrt(nsigma * sensitivity / (factor * gs[i]))

        print('nu, bound =', nu, bounds[i])

    np.savetxt('bounds_%s_%d.txt' % (telescope, ipulsar), np.c_[nus, bounds], fmt=['%.6e', '%.6e'], delimiter=' ', newline=os.linesep)

def make_forecast_collinear(ipulsar, telescope):
    nsigma = 2
    dicts = pulsar_dic[ipulsar]
    d_source = dicts['d_source']
    W =  dicts['W']
    P = dicts['P']
    timefactor = np.sqrt(W / (P-W))
    print('timefactor =', timefactor)

    if telescope == 'SKA1':
        gs = np.empty([len(nus_SKA1low) + len(nus_SKA1mid)])
        bounds = np.empty([len(nus_SKA1low) + len(nus_SKA1mid)])
        nus = np.concatenate((nus_SKA1low, nus_SKA1mid))
        beam_function = get_SKA1_beam
        sigma_function = get_SKA1_sigma

    if telescope == 'LOFAR':
        gs = np.empty([len(nus_LOFAR)])
        bounds = np.empty([len(nus_LOFAR)])
        nus = nus_LOFAR
        beam_function = get_LOFAR_beam
        sigma_function = get_LOFAR_sigma

    for i, nu in enumerate(nus):
        sensitivity = sigma_function(nu, deltanu_over_nu, 100) * 1e-6 * timefactor # Jy. nu must be in MHz. t must be in hours.
        flux_source = get_pulsar_flux(ipulsar)
        
        try:
            flux = flux_source(nu)
        except ValueError:
            print('### ValueError', nu)
            bounds[i] = 1
            continue
        
        deltanu = 2.8 * dvref * nu  # MHs
        factor =  fdeltanu / 16 / deltanu * d_source * flux  * 1.828e12
        
        gs[i] = integrate_los_collinear(dicts)
        bounds[i] = np.sqrt(nsigma * sensitivity / (factor * gs[i]))
        
        print()
        # print('beam =', beam)
        # print('sensitivity =', sensitivity)
        # print('factor =', factor)
        # print('flux =', flux)
        # print('gs[i, j] =', gs[i])
        print('nu, bound =', nu, bounds[i])
    
    np.savetxt('bounds_%s_%d_collinear.txt' % (telescope, ipulsar), np.c_[nus, bounds], fmt=['%.6e', '%.6e'],
               delimiter=' ', newline=os.linesep)
    #
    # for bound in bounds.flatten():
    #     print(bound)

#################################################
##              Plotting functions             ##
#################################################
def plot_sky_at_xd():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    
    
    t = 2  # t = xd/xs
    thetai_max = 5 * 2 * dv * (t + 1)  # radians
    thetais = np.linspace(-thetai_max, thetai_max, num=200)  # radians
    
    ax1.plot(thetais / arcmin, exponential_factor(thetad_behind_earth(thetais, t), dv), label=r'$x_d=%d\, x_s$' % t,
             color=colors[0], linestyle=':')
    ax2.plot(thetais / arcmin, exponential_factor(thetad_behind_source(thetais, t), dv), label=r'$x_d=%d\, x_s$' % t,
             color=colors[1], linestyle=':')
    #
    # thetad = np.arcsin(-np.sin(thetais) / np.sqrt(t ** 2 + 1 - 2 * t * np.cos(thetais)))
    # stuff = np.exp(- (thetad / dv)**2 / 2 / np.sqrt(4+thetad**2) ) / np.sqrt(1+thetad**2/4)
    # ax2.plot(thetais / arcmin, stuff, label=r'$x_d=%d\, x_s$' % t, color=colors[2], linestyle=':')
    #
    #
    # thetad = np.arcsin(-np.sin(thetais) / np.sqrt(t ** 2 + 1 + 2 * t * np.cos(thetais)))
    # stuff = np.exp(- (thetad / dv)**2 / 2 / np.sqrt(4+thetad**2) ) / np.sqrt(1+thetad**2/4)
    # ax1.plot(thetais / arcmin, stuff, label=r'$x_d=%d\, x_s$' % t, color=colors[2], linestyle=':')
    
    t = 3
    ax1.plot(thetais / arcmin, exponential_factor(thetad_behind_earth(thetais, t), dv), label=r'$x_d=%d\, x_s$' % t,
             color=colors[0])
    ax2.plot(thetais / arcmin, exponential_factor(thetad_behind_source(thetais, t), dv), label=r'$x_d=%d\, x_s$' % t,
             color=colors[1])
    
    fontsize = 22
    ax1.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax2.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$\mathrm{exp}\left(-\frac{(x_s\theta_i /x_{ ds })^2}{8\delta v^2}\right)$', fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=22)
    ax2.tick_params(axis='both', labelsize=22)
    
    ax1.set_title('Back-light echo', fontsize=fontsize)
    ax2.set_title('Front-light echo', fontsize=fontsize)
    
    ax1.legend(fontsize=18, frameon=False)
    ax2.legend(fontsize=18, frameon=False)
    # plt.show()
    plt.tight_layout()
    plt.savefig('sky_at_xd.pdf')
    
def plot_sky_integrated_los_analytical():
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), sharey=True, sharex=True)
    
    thetai_obs = 20 * arcmin  # radians
    
    # Plot idealized case
    thetais = np.logspace(-10, np.log10(thetai_obs), num=100)
    opposite = np.sqrt(2 * np.pi) * dv * (
        erf(thetais / (2 * np.sqrt(2) * dv))) / thetais
    ax1.plot(thetais / arcmin, opposite, label=r'Back-light echo', color=colors[0])
    ax1.plot(np.flip(-thetais) / arcmin, np.flip(opposite), color=colors[0])
    
    behind = np.sqrt(2 * np.pi) * dv / thetais
    ax1.plot(thetais / arcmin, behind, label=r'Front-light echo', color=colors[1])
    ax1.plot(np.flip(-thetais) / arcmin, np.flip(behind), color=colors[1])
    
    fontsize = 22
    ax1.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax1.set_ylabel(
        r'$g_{id}(\theta_i)$',
        fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=22)
    ax1.set_yscale('log')
    
    ax1.legend(fontsize=16, frameon=False)
    # plt.show()
    plt.tight_layout()
    plt.savefig('sky_integrated_los.pdf')
    
def plot_sky_integrated_los_halo_effect(isOpposite, isIdealizedSources):
    
    bigdict = idealized_sources_dic if isIdealizedSources else pulsar_dic
    filename = 'multi_isIdealizedSources=%s_isOpposite=%s_halo.pdf' % (str(isIdealizedSources), str(isOpposite))
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True, sharex=True)
    axesflat = axes.flatten()
    (ax1, ax2, ax3, ax4) = axesflat
    
    thetai_obs = 20 * arcmin  # radians
    
    # Plot analytical result for idealized case
    thetais = np.logspace(-10, np.log10(thetai_obs), num=200)
    
    if isOpposite:
        idealcase =  np.sqrt(2 * np.pi) * dvref / thetais * (
            erf(thetais / (2 * np.sqrt(2) * dvref)))
    else:
        idealcase = np.sqrt(2 * np.pi) * dvref / thetais
        print('idealcase =', idealcase[-1])

    for ax in axesflat:
        ax.plot(thetais / arcmin, idealcase, color='k', linewidth=2.5)
        ax.plot(np.flip(-thetais) / arcmin, np.flip(idealcase), color='k', linewidth=2.5)
        
    thetais = np.concatenate(([0], thetais))  # np.array([1]) * arcmin
    
    ipulsar_rangelimit = 4 if isIdealizedSources else 7
    ipulsar_range = range(ipulsar_rangelimit)
    ipulsar_range = range(4)
    
    for i_plot in range(4):
        
        for ipulsar in ipulsar_range:
            
            dicts = bigdict[ipulsar]
            l_source = dicts['l_source']
            b_source = dicts['b_source']
            name_source = dicts['name_source']

            # Frequency in units of axion mass
            if i_plot in [3, 8]:
                vpar, vperp, vy = get_vpar_vperp_vy(va, l_source, b_source, 0, 0, isOpposite)  # vpar for thetai=0
                omega = 0.5 * (1 + vpar)
            else:
                omega = 0.5
            
            gs = np.empty_like(thetais)
            print('\n*****************', name_source, isOpposite, i_plot)
            for i, thetai in enumerate(thetais):
                gs[i] = integrate_los(thetai, 0, dicts, omega, i_plot, isOpposite, isIdealizedSources)
                
            # print(thetais[-1]/arcmin, gs[-1])
            axesflat[i_plot].plot(thetais / arcmin, gs, label=name_source, color=colors[ipulsar])
            
            for i, thetai in enumerate(thetais):
                gs[i] = integrate_los(thetai, pi, dicts, omega, i_plot, isOpposite, isIdealizedSources)
            axesflat[i_plot].plot(np.flip(-thetais) / arcmin, np.flip(gs), color=colors[ipulsar])
            
            axesflat[i_plot].tick_params(axis='both', labelsize=18)
    
    fontsize = 18
    ax3.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax4.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$g(\theta_i)$', fontsize=fontsize)
    ax3.set_ylabel(r'$g(\theta_i)$', fontsize=fontsize)

    
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_yscale('log')
    
    myy = 1.05 # 0.1 if isOpposite else 0.8
    myx = 0.5
    mypad = 0  # -12
    ax1.set_title(r'$\rho(r)$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax2.set_title(r'$\rho(r),~x_s$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax3.set_title(r'$\rho(r),~x_s,~\delta v(r)$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax4.set_title(r'$\langle \vec{v}\,\rangle$', y=myy, x=myx, pad=mypad, fontsize=20)

    
    if not isOpposite:
        ax1.set_ylim(top=1e5)  # (d_source/radius_source / 100))
    ax1.set_ylim(bottom=2e-2)  # (d_source/radius_source / 100))
    location = 'lower center' if isOpposite else 'upper right'
    ax1.legend(fontsize=14, loc=location, frameon=False)
    # ax4.legend(fontsize=14)
    # ax4.set_xscale('log')
    # ax1.set_xlim([-21,21])
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def plot_sky_integrated_los_source_effect(isOpposite, isIdealizedSources):

    bigdict = idealized_sources_dic if isIdealizedSources else pulsar_dic
    filename = 'multi_isIdealizedSources=%s_isOpposite=%s_source.pdf' % (str(isIdealizedSources), str(isOpposite))
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True, sharex=True)
    axesflat = axes.flatten()
    (ax1, ax2, ax3, ax4) = axesflat
    
    thetai_obs = 20 * arcmin  # radians
    
    # Plot analytical result for idealized case
    thetais = np.logspace(-10, np.log10(thetai_obs), num=200)
    
    if isOpposite:
        idealcase =  np.sqrt(2 * np.pi) * dvref / thetais * (
            erf(thetais / (2 * np.sqrt(2) * dvref)))
    else:
        idealcase = np.sqrt(2 * np.pi) * dvref / thetais
        print('idealcase =', idealcase[-1])

    for ax in axesflat:
        ax.plot(thetais / arcmin, idealcase, color='k', linewidth=2.5)
        ax.plot(np.flip(-thetais) / arcmin, np.flip(idealcase), color='k', linewidth=2.5)
    
    thetais = np.concatenate(([0], thetais))  # np.array([1]) * arcmin

    ipulsar_rangelimit = 4 if isIdealizedSources else 7
    ipulsar_range = range(ipulsar_rangelimit)
    # ipulsar_range = range(4)

    for i_plot in range(4,8):
    
        for ipulsar in ipulsar_range:
            dicts = bigdict[ipulsar]
            l_source = dicts['l_source']
            b_source = dicts['b_source']
            name_source = dicts['name_source']

            # Frequency in units of axion mass
            if i_plot in [3, 8]:
                vpar, vperp, vy = get_vpar_vperp_vy(va, l_source, b_source, 0, 0, isOpposite)  # vpar for thetai=0
                omega = 0.5 * (1 + vpar)
            else:
                omega = 0.5
            
            gs = np.empty_like(thetais)
            print('\n*****************', name_source, isOpposite, i_plot)
            for i, thetai in enumerate(thetais):
                gs[i] = integrate_los(thetai, 0, dicts, omega, i_plot, isOpposite, isIdealizedSources)

            axesflat[i_plot-4].plot(thetais / arcmin, gs, label=name_source, color=colors[ipulsar])
            
            for i, thetai in enumerate(thetais):
                gs[i] = integrate_los(thetai, pi, dicts, omega, i_plot, isOpposite, isIdealizedSources)
            axesflat[i_plot-4].plot(np.flip(-thetais) / arcmin, np.flip(gs), color=colors[ipulsar])
            
            axesflat[i_plot-4].tick_params(axis='both', labelsize=18)
    
    fontsize = 18
    ax3.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)
    ax4.set_xlabel(r'$\theta_i~[\mathrm{arcmin}]$', fontsize=fontsize)

    ax1.set_ylabel(r'$g(\theta_i)$', fontsize=fontsize)
    ax3.set_ylabel(r'$g(\theta_i)$', fontsize=fontsize)

    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_yscale('log')
    
    myy = 1.05 # 0.1 if isOpposite else 0.8
    myx = 0.5
    mypad = 0  # -12
    ax1.set_title(r'$t_{age}$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax2.set_title(r'$t_{age},~x_s$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax3.set_title(r'$t_{age},~x_s,~L_\nu(t)$', y=myy, x=myx, pad=mypad, fontsize=20)
    ax4.set_title(r'$\vec{v}_s$', y=myy, x=myx, pad=mypad, fontsize=20)
    
    if not isOpposite:
        ax1.set_ylim(top=1e5)  # (d_source/radius_source / 100))
    ax1.set_ylim(bottom=2e-2)  # (d_source/radius_source / 100))
    location = 'lower center' if isOpposite else 'upper right'
    ax1.legend(fontsize=14, loc=location, frameon=False)
    # ax4.legend(fontsize=14)
    # ax4.set_xscale('log')
    # ax1.set_xlim([-21,21])
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def get_intensity_of_theta_2D():
    thetai_obs = 20 * arcmin
    xs = np.linspace(-thetai_obs, thetai_obs, num=200)
    ys = np.linspace(-thetai_obs, thetai_obs, num=201)

    X, Y = np.meshgrid(xs, ys)
    thetais = np.sqrt(X ** 2 + Y ** 2)
    phiis = np.arctan2(X, Y) # phii=0 means positive galactic latitude
    # print(phiis)

    ipulsar_range = range(4)
    
    for ipulsar in ipulsar_range:
        
        dicts = pulsar_dic[ipulsar]
        d_source = dicts['d_source']
        s400 = get_pulsar_flux(ipulsar)(400)  # dicts['s400']
        print('flux =', s400)
        name_source = dicts['name_source']
        #
        g = 10 ** -11  # GeV^-1
        deltanu = 2.8 * 400 * dvref  # MHs
        factor = fdeltanu * g ** 2 / 32 / deltanu * d_source * s400 / (
                2 * pi) * 1.828e12  # conversion factor for GeV^-2/MHz kpc GeV/cm^3
        
        gs = np.empty_like(X)
        # gs = np.empty_like(thetais_c)
        
        print('\n*****************', name_source)
        for i in range(len(thetais)):
            j=0
            for thetai, phii in zip(thetais[i], phiis[i]):
                # if ipulsar==3:
                #     print('\nX, Y=', X[i,j]/arcmin, Y[i,j]/arcmin)
                #     print('thetai, phii =', thetai, phii * 180 / pi)
                
                gs[i, j] = integrate_los_for_flux(thetai, phii, ipulsar, dicts, isOpposite=False)
                # if ipulsar==3:
                # if gs[i, j]*factor < 1e-23:
                #     print('thetai, =', X[i,j]/arcmin, Y[i,j]/arcmin, gs[i, j]*factor)
                j += 1

        gs = factor * gs

        np.savez('intensity_2D_%d.npz' % ipulsar, gs=gs, xs=xs, ys=ys)
        
def fmt(x):
    exponent = np.log10(x)
    # s = f"{x:.1f}"
    # if s.endswith("0"):
    #     s = f"{x:.0f}"
    # return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
    return r'$10^{%d}$' % exponent

def format_func(value, tick_number):

    if value <= 0:
        return r"$%d$" % np.abs(value)
    else:
        return r"$-%d$" % np.abs(value)

def plot_intensity_of_theta_2D():
    filename = 'sky_intensity_2D.pdf'

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), sharey=True, sharex=True)
    axes = axes.flatten()
    big_gs = []
    myrange = range(4)
    for ipulsar in myrange:
        
        data = np.load('intensity_2D_%d.npz' % ipulsar)
        big_gs.append(data['gs'])
        xs = data['xs']
        ys = data['ys']
    

    big_gs = np.array(big_gs)
    maxg = np.amax(big_gs)
    ming = np.amin(big_gs)
    
    print('ming =', ming)
    print('maxg =', maxg)
    
    big_gs[np.where(big_gs<ming)] = np.nan

    norm = colorss.LogNorm(vmin=ming, vmax=maxg)
    print(norm.vmin, norm.vmax)
    for ax, ipulsar in zip(axes, myrange):
        # Get DM velocity components
        
        dict = pulsar_dic[ipulsar]
        vx, vy, vz = np.array(get_velocity_components_in_source_system(va, dict['l_source'], dict['b_source'], 0,
                                                              isOpposite=False)) * 1e4
    
        # Get source velocity
        vs_x, vs_y, vs_z = np.array(get_velocity_components_in_source_system(dict['vs'], dict['l_source'], dict['b_source'], 0,
                                                                    isOpposite=False) )* 1e4
    
        np.set_printoptions(suppress=True)
        
        X, Y = np.meshgrid(xs, ys)
        
        # im = ax.pcolormesh(Y / arcmin, X / arcmin, big_gs[ipulsar], norm=colorss.LogNorm(vmin=min, vmax=max))
        extent = [xs.min() / arcmin, xs.max() / arcmin, ys.min() / arcmin, ys.max() / arcmin]
        # extent = [-20, 20, -20, 21]
        im = ax.imshow(big_gs[ipulsar], interpolation='nearest', norm=norm,
                       extent=extent, origin='lower', cmap='rainbow')
        #
        CP = ax.contour(X/arcmin, Y/arcmin, big_gs[ipulsar] , levels=[1e-7, 1e-6, 1e-5], linewidths=1.0, colors='k') #1e-8, 1e-6, 1e-4,
        ax.clabel(CP, CP.levels, inline=True, fmt=fmt, fontsize=14)
        
        
        ax.arrow(0, 0, dx=vy, dy=vx, width=0.1, head_width=1, head_length=1, fc='b', ec='b', color='g')
        if vs_x != 0:
            ax.arrow(0, 0, dx=vs_y, dy=vs_x, width=0.1, head_width=1, head_length=1, fc='magenta', ec='magenta')
        myy = 0.91
        myx = 0.3
        mypad = 0  # -12
        # ax.set_title(r'$\mathrm{%s}$' % pulsar_dic[ipulsar]['name_source'], x=myx,  y=myy, fontsize=20)
        ax.set_title(pulsar_dic[ipulsar]['name_source'], x=myx,  y=myy, fontsize=22)

        ax.grid(False)
        ax.set_aspect(1)
        xtics = np.flip(range(-20, 25, 10))
        ax.set_xticks(xtics)
        ax.set_xticklabels(xtics)


        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.11, hspace=0.04)

    fontsize = 22
    cbar.set_label(r'$\langle I_\nu \rangle~[\mathrm{Jy~sr}^{-1}]$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=25)
    
    axes[2].set_xlabel(r'$\Delta l~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[3].set_xlabel(r'$\Delta l~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[0].set_ylabel(r'$\Delta b~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[2].set_ylabel(r'$\Delta b~[\mathrm{arcmin}]$', fontsize=fontsize)

    axes[0].tick_params(axis='both', labelsize=22)
    axes[2].tick_params(axis='both', labelsize=22)
    axes[3].tick_params(axis='both', labelsize=22)

    # plt.tight_layout()
    # plt.show()
    plt.savefig(filename, bbox_inches='tight')

def plot_intensity_of_theta_behind():
    filename = 'sky_intensity_behind.pdf'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    
    thetai_obs = 20 * arcmin  # radians
    
    phiis = np.array([0, pi])
    # phiis = np.array([60*degree, 60*degree + pi])
    
    rangelimit = 8
    myrange = range(rangelimit)
    myrange = range(4)
    
    iii = 0
    for ax, phiis in zip(axes, [np.array([0, pi]), np.array([pi / 2, 3 * pi / 2])]):
        # logthetamin = -5 if isOpposite else -14
        logthetamin = -14
        thetais = np.logspace(logthetamin, np.log10(thetai_obs), num=200)  # center of the beam # -12,
        for ipulsar in myrange:
            dicts = pulsar_dic[ipulsar]
            d_source = dicts['d_source']
            s400 = get_pulsar_flux(ipulsar)(400)  # dicts['s400']
            print('flux =', s400)
            name_source = dicts['name_source']
            #
            g = 10 ** -11  # GeV^-1
            fdeltanu = 0.84
            deltanu = 2.8 * 400 * dvref  # MHs
            factor = fdeltanu * g ** 2 / 32 / deltanu * d_source * s400 / (
                        2 * pi) * 1.828e12  # conversion factor for GeV^-2/MHz kpc GeV/cm^3
            
            gs = np.empty([len(thetais), len(phiis)])
            # gs = np.empty_like(thetais_c)
            
            print('\n*****************', name_source)
            for j, phii in enumerate(phiis):
                for i, thetai in enumerate(thetais):
                    gs[i, j] = integrate_los_for_flux(thetai, phii, ipulsar, dicts, isOpposite=False)
                    # if ipulsar==0 and isOpposite:
                    #     print(thetai/arcmin, gs[i, j])
            if iii == 0:
                print(ipulsar, iii)
                ax.plot(thetais / arcmin, factor * gs[:, 0], label=pulsar_dic[ipulsar]['name_source'],
                        color=colors[ipulsar])
                ax.plot(np.flip(-thetais) / arcmin, factor * np.flip(gs[:, 1]), color=colors[ipulsar])
            else:  # flip because we're plotting \Delta l, which has the opposite direction of the y axis
                print(ipulsar, iii)
                
                ax.plot(thetais / arcmin, factor * gs[:, 1], label=pulsar_dic[ipulsar]['name_source'],
                        color=colors[ipulsar])
                ax.plot(np.flip(-thetais) / arcmin, factor * np.flip(gs[:, 0]), color=colors[ipulsar])
        iii += 1
    
    fontsize = 22
    axes[0].set_xlabel(r'$\Delta b~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[1].set_xlabel(r'$\Delta l~[\mathrm{arcmin}]$', fontsize=fontsize)
    # axes[1].invert_xaxis()
    # axes[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # xtics = np.flip(range(-20, 25, 10))
    # axes[1].set_xticks(xtics)
    # axes[1].set_xticklabels(xtics)
    
    axes[0].set_ylabel(r'$\langle I_\nu \rangle~[\mathrm{Jy~sr}^{-1}]$', fontsize=fontsize)
    
    axes[0].tick_params(axis='both', labelsize=22)
    axes[1].tick_params(axis='both', labelsize=22)
    axes[0].set_yscale('log')
    
    #
    axes[1].set_ylim(top=1e-1)  # (d_source/radius_source / 100))
    axes[0].set_ylim(bottom=1e-10)  # (d_source/radius_source / 100))
    
    # location = 'lower center' if isOpposite else 'upper right'
    axes[1].legend(fontsize=18, loc='upper right', frameon=False)
    # ax4.legend(fontsize=14)
    # ax4.set_xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def plot_intensity_of_theta_behind_opposite():
    filename = 'sky_intensity_behind_opposite.pdf'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    
    thetai_obs = 20 * arcmin  # radians
    
    phiis = np.array([0, pi])
    # phiis = np.array([60*degree, 60*degree + pi])
    
    rangelimit = 8
    myrange = range(rangelimit)
    myrange = range(4)
    
    for ax, isOpposite in zip(axes, [True, False]):
        logthetamin = -5 if isOpposite else -14
        thetais = np.logspace(logthetamin, np.log10(thetai_obs), num=200)  # center of the beam # -12,
        for ipulsar in myrange:
            dicts = pulsar_dic[ipulsar]
            d_source = dicts['d_source']
            s400 = get_pulsar_flux(ipulsar)(400)  # dicts['s400']
            print('flux =', s400)
            name_source = dicts['name_source']
            #
            g = 10 ** -11  # GeV^-1
            fdeltanu = 0.84
            deltanu = 2.8 * 400 * dvref  # MHs
            factor = fdeltanu * g ** 2 / 32 / deltanu * d_source * s400 / (
                    2 * pi) * 1.828e12  # conversion factor for GeV^-2/MHz kpc GeV/cm^3
            
            gs = np.empty([len(thetais), len(phiis)])
            # gs = np.empty_like(thetais_c)
            
            print('\n*****************', name_source)
            for j, phii in enumerate(phiis):
                for i, thetai in enumerate(thetais):
                    gs[i, j] = integrate_los_for_flux(thetai, phii, ipulsar, dicts, isOpposite=isOpposite)
                    # if ipulsar==0 and isOpposite:
                    #     print(thetai/arcmin, gs[i, j])
            
            ax.plot(thetais / arcmin, factor * gs[:, 0], label=name_source, color=colors[ipulsar])
            ax.plot(np.flip(-thetais) / arcmin, factor * np.flip(gs[:, 1]), color=colors[ipulsar])
    
    fontsize = 22
    axes[0].set_xlabel(r'$\Delta b~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[1].set_xlabel(r'$\Delta l~[\mathrm{arcmin}]$', fontsize=fontsize)
    axes[0].set_ylabel(r'$\langle I_\nu \rangle~[\mathrm{Jy~sr}^{-1}]$', fontsize=fontsize)
    
    axes[0].tick_params(axis='both', labelsize=22)
    axes[1].tick_params(axis='both', labelsize=22)
    axes[0].set_yscale('log')
    
    #
    axes[1].set_ylim(top=1e-1)  # (d_source/radius_source / 100))
    axes[0].set_ylim(bottom=1e-10)  # (d_source/radius_source / 100))
    
    # location = 'lower center' if isOpposite else 'upper right'
    axes[1].legend(fontsize=18, loc='upper right', frameon=False)
    # ax4.legend(fontsize=14)
    # ax4.set_xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def plot_existing_limits(ax):
    
    limitdir = 'limits/'
    # # print(os.listdir('limits/haloscopes'))
    # for i, filename in enumerate(os.listdir(limitdir + 'haloscopes/')):
    #     data = np.loadtxt(limitdir+ 'haloscopes/' + filename)
    #     mass = data[:,0]
    #     bound = data[:, 1]
    #     if i ==0:
    #         ax.fill_between(mass, bound, 1, alpha=0.2, color='lightsalmon')#, label='Haloscopes')
    #     else:
    #         ax.fill_between(mass, bound, 1, alpha=0.2, color='r')

    data = np.loadtxt(limitdir + 'Haloscopes_Combined_microeV-meV.txt')
    mass = data[:, 0]
    bound = data[:, 1]
    ax.fill_between(mass, bound, 1, alpha=0.3, color='lightsalmon')
    ax.text(3e-6, 2e-12, 'Haloscopes', fontsize=18, color='darkred')

    data = np.loadtxt(limitdir + 'CAST.txt')
    mass = data[:, 0]
    bound = data[:, 1]
    
    boundCAST = interp1d(mass, bound, bounds_error=False, fill_value='extrapolate')
    # ax.fill_between(mass, bound, 1, alpha=0.1, color='limegreen', label='CAST', zorder=-200)
    ax.plot(mass, bound,  alpha=1, color='g')#, label='CAST')
    ax.text(1.3e-7, 8e-11, 'CAST', fontsize=18, color='g')

    
    data1 = np.loadtxt(limitdir + 'GlobularClusters-R2.txt')
    mass1 = [1.e-8, 1.e-3] #data1[:, 0]
    bound1 = data1[:, 1]

    ax.fill_between(mass1, bound1,1, alpha=0.1, color='limegreen', zorder=-300)
    ax.plot(mass1, bound1, alpha=1, color='darkgreen')#, label='GC')
    ax.text(1.3e-7, 2.61e-11, 'Globular clusters', fontsize=18, color='darkgreen')
    # print(bound)
    
def plot_bounds():
    
    filename = 'forecast.pdf'
    
    nutom =  8.27119e-9
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharey=True, sharex=True)

    plot_existing_limits(ax)
    
    for i in [0,2,3]:
        data = np.loadtxt('bounds_SKA1_%d.txt' % i)
        mass= data[:, 0] * nutom # eV
        bounds = data[:, 1]
        name_source = pulsar_dic[i]['name_source']
        bounds[np.where(bounds>1e-1)] = np.nan
    
        ax.plot(mass , bounds, color=colors[i], linestyle='--')

    i = 1
    data = np.loadtxt('bounds_LOFAR_%d.txt' % i)
    mass = data[:, 0] * nutom  # eV
    bounds = data[:, 1]
    name_source = pulsar_dic[i]['name_source']
    bounds[np.where(bounds > 1e-1)] = np.nan
    ax.plot(mass, bounds, color=colors[i], linestyle='--')

    for i in [0, 2, 3]:
        data = np.loadtxt('bounds_SKA1_%d_collinear.txt' % i)
        mass = data[:, 0] * nutom # eV
        bounds = data[:, 1]
        name_source = pulsar_dic[i]['name_source']
        bounds[np.where(bounds > 1e-1)] = np.nan
    
        ax.plot(mass, bounds, color=colors[i], label=name_source)
        

    i = 1
    data = np.loadtxt('bounds_LOFAR_%d_collinear.txt' % i)
    mass = data[:, 0] * nutom  # eV
    bounds = data[:, 1]
    name_source = pulsar_dic[i]['name_source']
    bounds[np.where(bounds > 1e-1)] = np.nan
    ax.plot(mass, bounds, color=colors[i], label=name_source)


    i = 0
    data = np.loadtxt('bounds_SKA1_%d_collinear.txt' % i)
    mass = data[:, 0] * nutom  # eV
    bounds = data[:, 1]
    name_source = pulsar_dic[i]['name_source']
    bounds[np.where(bounds > 1e-1)] = np.nan

    ax.plot(mass, bounds / 10.2, color='k', linestyle='--', label='SKA-1 survey')
    ax.plot(mass, bounds / 10.2 / 10, color='k', label='SKA-2 survey')
   
   
   
   
    fontsize = 22
    ax.set_xlabel(r'$m_a~[\mathrm{eV}]$', fontsize=fontsize)
    ax.set_ylabel(r'$g_{a\gamma}~[\mathrm{GeV}^{-1}]$', fontsize=fontsize)
    
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_ylim([1e-12, 3e-4])
    ax.set_xlim([1e-7, 1e-3])

    location = 'upper left'
    ax.legend(fontsize=15,  frameon=False, loc=location)
    # ax4.legend(fontsize=14)
    # ax4.set_xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    

#################################################
#################################################
#################################################
if __name__ == '__main__':
    rho0, r0 = get_NFW_parameters()
    rhosun = rhoNFW(rsun)
    
    # Default values
    vperpref = 0.0
    xsref = 0.1 # kpc
    rstar = 3.24078e-16 # kpc
    rhoref = rhosun
    dvref = get_dv(rsun)[0]
    print('dvref =', dvref, dvref * c)
    xsref = 0.1 # kpc
    deltanu_over_nu = 2.8 * dvref

    plot_sky_at_xd() # Figure 3
    plot_sky_integrated_los_analytical() # Figure 4
    
    plot_sky_integrated_los_halo_effect(isOpposite=True, isIdealizedSources=True) # Figure 5
    plot_sky_integrated_los_halo_effect(isOpposite=False, isIdealizedSources=True) # Figure 6
    plot_sky_integrated_los_source_effect(isOpposite=True, isIdealizedSources=True) # Figure 7
    plot_sky_integrated_los_source_effect(isOpposite=False, isIdealizedSources=True) # Figure 8

    plot_sky_integrated_los_halo_effect(isOpposite=True, isIdealizedSources=False) # Figure 5, but for real sources
    plot_sky_integrated_los_halo_effect(isOpposite=False, isIdealizedSources=False) # Figure 6, but for real sources
    plot_sky_integrated_los_source_effect(isOpposite=True, isIdealizedSources=False) # Figure 7, but for real sources
    plot_sky_integrated_los_source_effect(isOpposite=False, isIdealizedSources=False) # Figure 8, but for real sources

    get_intensity_of_theta_2D() # creates the data for Figure 9
    plot_intensity_of_theta_2D() # Figure 9
    
    plot_intensity_of_theta_behind() # Figure 10
    plot_intensity_of_theta_behind_opposite()  # Figure 10, but for backlight echo
    
    make_forecast(ipulsar=0, telescope='SKA1')
    make_forecast(ipulsar=2, telescope='SKA1')
    make_forecast(ipulsar=3, telescope='SKA1')
    make_forecast(ipulsar=1, telescope='LOFAR')
    make_forecast_collinear(ipulsar=0, telescope='SKA1')
    make_forecast_collinear(ipulsar=2, telescope='SKA1')
    make_forecast_collinear(ipulsar=3, telescope='SKA1')
    make_forecast_collinear(ipulsar=1, telescope='LOFAR')

    plot_bounds() # Figure 11
