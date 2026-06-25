#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║    NIOT — USBL + BELLHOP INTEGRATED ACOUSTIC PROPAGATION        ║
║    Real-Time Underwater Communication System                     ║
║    Supports: Shallow Water (Pond) ↔ Deep Water (Coastal)        ║
║                                                                  ║
║    Bellhop Output Files Supported:                               ║
║      .env  – Environment input file                              ║
║      .prt  – Print/log output                                    ║
║      .ray  – Ray trace data                                      ║
║      .shd  – Transmission loss shade file                        ║
║      .arr  – Arrival data (eigenrays)                            ║
║                                                                  ║
║    Acoustic Losses Modelled:                                     ║
║      • Spherical & Cylindrical geometric spreading               ║
║      • Absorption (Francois-Garrison formula)                    ║
║      • Surface scattering (wind/wave dependent)                  ║
║      • Bottom reflection loss (sediment type)                    ║
║      • Volume scattering (biological / bubble layers)            ║
║      • Multipath interference penalty                            ║
║      • Ambient noise (Wenz curves)                               ║
║      • Doppler shift (moving platforms)                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from queue import Queue
import socket
import zlib
import time
import datetime
import numpy as np
import os
import subprocess
import math
import struct
import tempfile
import shutil
from pymavlink import mavutil
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════
#  ACOUSTIC PHYSICS — SOUND SPEED EQUATIONS
# ══════════════════════════════════════════════════════════════════

def sound_speed_seawater(T, S, D):
    """
    UNESCO/Chen-Millero sound speed in seawater.
    T: temperature (°C), S: salinity (ppt), D: depth (m)
    Accuracy: ±0.1 m/s
    """
    c = (1449.2 + 4.6*T - 0.055*T**2 + 0.00029*T**3
         + (1.34 - 0.010*T)*(S - 35) + 0.016*D)
    return c

def sound_speed_freshwater(T, D=0.0):
    """
    Medwin (1975) formula for fresh water.
    T: temperature (°C), D: depth (m)
    """
    c = (1402.5 + 5.0*T - 0.05585*T**2 + 0.000339*T**3 + 0.00071*D)
    return c

def ssp_shallow_pond(depth_max=3.0, T_surface=28.0, n_pts=30):
    """
    Sound Speed Profile for a fresh-water pond (Chennai ~28°C).
    Near-isothermal with slight depth-pressure gradient.
    """
    depths = np.linspace(0.0, depth_max, n_pts)
    T_profile = T_surface - 0.5 * (depths / depth_max)   # mild gradient
    speeds = np.array([sound_speed_freshwater(T, d) for T, d in zip(T_profile, depths)])
    return depths, speeds

def ssp_coastal_deep(depth_max=80.0, T_surface=28.0, S=35.0, n_pts=60):
    """
    Sound Speed Profile for Bay of Bengal coastal environment.
    Thermocline ~20 m; below: gradual decrease then slight pressure increase.
    """
    depths = np.linspace(0.0, depth_max, n_pts)
    speeds = []
    for d in depths:
        # Exponential thermocline
        T = T_surface * math.exp(-d / 25.0) + 5.0
        c = sound_speed_seawater(T, S, d)
        speeds.append(c)
    return depths, np.array(speeds)


# ══════════════════════════════════════════════════════════════════
#  ACOUSTIC PHYSICS — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def francois_garrison_absorption(f_Hz, T, S, D, pH=8.0):
    """
    Francois & Garrison (1982) seawater absorption formula.
    Three relaxation terms: Boric acid, MgSO₄, pure-water viscosity.
    f_Hz: frequency (Hz), T: temp (°C), S: salinity (ppt), D: depth (m)
    Returns: absorption coefficient α (dB/m)
    """
    f = f_Hz / 1000.0           # work in kHz internally
    c = sound_speed_seawater(T, S, D)

    # ── Boric acid term (low f) ──────────────────────────────────
    A1 = (8.86 / c) * 10 ** (0.78 * pH - 5.0)
    f1 = 2.8 * math.sqrt(S / 35.0) * 10 ** (4.0 - 1245.0 / (T + 273.0))
    P1 = 1.0

    # ── MgSO₄ term (mid f) ─────────────────────────────────────
    A2 = 21.44 * (S / c) * (1.0 + 0.025 * T)
    f2 = 8.17 * 10 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))
    P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D**2

    # ── Pure water viscosity term (high f) ──────────────────────
    if T <= 20.0:
        A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T**2 - 1.5e-8*T**3
    else:
        A3 = 3.964e-4 - 1.146e-5*T + 1.45e-7*T**2 - 6.5e-10*T**3
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D**2

    alpha_dB_km = (A1 * P1 * f1 * f**2 / (f1**2 + f**2)
                   + A2 * P2 * f2 * f**2 / (f2**2 + f**2)
                   + A3 * P3 * f**2)

    return alpha_dB_km / 1000.0     # → dB/m

def thorp_absorption(f_Hz):
    """
    Thorp (1967) simplified absorption, good for f < 100 kHz.
    Returns dB/km.
    """
    f = f_Hz / 1000.0
    return (0.11 * f**2 / (1.0 + f**2)
            + 44.0 * f**2 / (4100.0 + f**2)
            + 2.75e-4 * f**2 + 0.003)

def geometric_spreading_loss(r_m, mode='spherical'):
    """
    Geometric (propagation) spreading loss.
    mode: 'spherical' (20logR), 'cylindrical' (10logR),
          'practical' (Marsh-Schulkin mixed-law)
    """
    r = max(r_m, 0.01)
    if mode == 'spherical':
        return 20.0 * math.log10(r)
    elif mode == 'cylindrical':
        return 10.0 * math.log10(r)
    else:       # practical mixed (transition at ~1000 m)
        if r <= 1000.0:
            return 20.0 * math.log10(r)
        return 20.0 * math.log10(1000.0) + 10.0 * math.log10(r / 1000.0)

def surface_scattering_loss(f_Hz, wind_ms, grazing_deg):
    """
    Surface scattering loss using Chapman-Harris (1962) / Eckart model.
    Rayleigh roughness parameter R = 4π σ sinθ / λ
    Returns loss in dB.
    """
    if grazing_deg < 0.5:
        grazing_deg = 0.5
    theta = math.radians(grazing_deg)
    c     = 1500.0
    lam   = c / f_Hz          # wavelength (m)

    # RMS wave height (simplified Pierson-Moskowitz for short fetch)
    sigma_h = 0.0248 * wind_ms**1.5 / math.sqrt(f_Hz / 1000.0 + 1.0)
    sigma_h = max(sigma_h, 0.001)

    R = 4.0 * math.pi * sigma_h * math.sin(theta) / lam
    rho_s = math.exp(-(R**2))          # specular reflection coefficient
    return -20.0 * math.log10(max(rho_s, 1e-12))

def bottom_reflection_loss(f_Hz, grazing_deg, bottom_type='sand'):
    """
    Bottom reflection loss using plane-wave Rayleigh reflection coefficient.
    Returns BL in dB (positive = loss).

    Bottom properties table (practical measured values):
      cp  : compressional speed (m/s)
      cs  : shear speed (m/s)
      rho : sediment density (g/cc)
      αp  : compressional attenuation (dB/wavelength)
      αs  : shear attenuation (dB/wavelength)
    """
    props = {
        'sand':  dict(cp=1650, cs=200,  rho=1.90, ap=0.80, as_=2.5),
        'mud':   dict(cp=1520, cs=50,   rho=1.50, ap=1.50, as_=1.0),
        'clay':  dict(cp=1550, cs=80,   rho=1.60, ap=1.20, as_=1.5),
        'rock':  dict(cp=3500, cs=2000, rho=2.30, ap=0.10, as_=0.2),
        'gravel':dict(cp=1800, cs=400,  rho=2.00, ap=0.60, as_=2.0),
        'silt':  dict(cp=1530, cs=60,   rho=1.55, ap=1.30, as_=1.2),
    }
    p   = props.get(bottom_type, props['sand'])
    c_w = 1500.0        # water sound speed
    rho_w = 1025.0      # water density (kg/m³)

    theta = math.radians(max(grazing_deg, 0.1))
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    # Snell's law — complex bottom angle
    n = c_w / p['cp']
    sin_tb_sq = cos_t**2            # sin²(θ_bottom) from Snell
    cos_tb = math.sqrt(max(0.0, 1.0 - sin_tb_sq * n**2))

    # Impedances
    Z_w  = rho_w * c_w / sin_t
    Z_b  = (p['rho'] * rho_w * p['cp']) / max(cos_tb, 1e-9)

    V = (Z_b - Z_w) / (Z_b + Z_w)
    return -20.0 * math.log10(max(abs(V), 1e-12))

def volume_scattering_loss(f_Hz, range_m, depth_m):
    """
    Volume backscattering loss from biological scattering layers (DSL).
    Uses empirical Sv ≈ −75 dB/m³ at 25 kHz (typical mid-water).
    Returns extra TL in dB.
    """
    # Deep scattering layer strength (frequency-dependent)
    f_kHz = f_Hz / 1000.0
    Sv    = -75.0 + 10.0 * math.log10(max(f_kHz / 25.0, 0.1))
    # Reverberation level (one-way): RL ≈ Sv + 10log(V_scatter)
    V_cell = max(range_m * depth_m * 0.1, 1.0)  # approximate scatter volume
    RL     = Sv + 10.0 * math.log10(V_cell)
    return max(-RL / 10.0, 0.0)    # small extra loss in practice

def bubble_attenuation(f_Hz, depth_m, bubble_density=1e3):
    """
    Near-surface bubble layer extra attenuation (Thorpe 1982).
    bubble_density: number per m³ (default light bubble layer).
    Significant only in rough sea conditions.
    Returns extra absorption dB/m.
    """
    f = f_Hz
    r_b = 100e-6                    # typical bubble radius 100 µm
    omega = 2.0 * math.pi * f
    # resonance frequency of bubble
    f_res = (1.0 / (2.0 * math.pi * r_b)) * math.sqrt(3.0 * 1.4 * 101325.0 / 1000.0)
    Q = 10.0                        # quality factor
    sigma_ext = (4.0 * math.pi * r_b**2 * (f_res / f)**2
                 / ((1.0 - (f / f_res)**2)**2 + (1.0 / Q)**2))
    alpha_bubble = 10.0 * math.log10(math.e) * bubble_density * sigma_ext
    return alpha_bubble

def doppler_shift(f0, v_src_ms, v_rcv_ms, c):
    """
    Relativistic-free Doppler shift.
    v_src, v_rcv: radial velocities (positive = moving away from each other)
    Returns observed frequency (Hz).
    """
    return f0 * (c + v_rcv_ms) / (c + v_src_ms)

def ambient_noise_wenz(f_Hz, sea_state=2, shipping_level=5):
    """
    Wenz (1962) ambient noise spectrum.
    Returns noise spectral density NL (dB re 1 µPa²/Hz).
    """
    f_kHz = f_Hz / 1000.0
    # Shipping noise
    s     = max(0, min(10, shipping_level))
    N_s   = 76.0 - 20.0 * math.log10(f_kHz) + (s - 5) * 5.0

    # Wind/surface noise
    w     = max(0, min(9, sea_state))
    N_w   = 44.0 + 7.5 * math.sqrt(w) + 20.0 * math.log10(f_kHz) * (-1 if f_kHz > 1 else 1)
    N_w   = 44.0 + 7.5 * math.sqrt(w) - 20.0 * math.log10(f_kHz)

    # Thermal noise (dominant >50 kHz)
    N_th  = -15.0 + 20.0 * math.log10(f_kHz)

    # Turbulence (<10 Hz)
    N_t   = 17.0 - 30.0 * math.log10(f_kHz)

    # Combine incoherently
    total = 10.0 * math.log10(
        10**(N_s/10) + 10**(N_w/10) + 10**(N_th/10)
    )
    return total

def total_transmission_loss(r_m, f_Hz, T, S, D_water,
                             src_depth, rec_depth,
                             bottom_type, wind_ms=5.0, sea_state=2):
    """
    Comprehensive TL budget.
    Returns dict with all components (dB) and totals.
    """
    r = max(r_m, 0.1)

    # 1. Geometric spreading (practical mixed law)
    TL_geo_sph = 20.0 * math.log10(r)
    TL_geo_cyl = 10.0 * math.log10(r)
    TL_geo     = geometric_spreading_loss(r, 'practical')

    # 2. Absorption (Francois-Garrison)
    alpha_FG = francois_garrison_absorption(f_Hz, T, S, D_water)   # dB/m
    TL_abs   = alpha_FG * r

    # 3. Surface scattering
    grazing = math.degrees(math.atan(D_water / r)) if r > 0 else 45.0
    TL_surf  = surface_scattering_loss(f_Hz, wind_ms, grazing)

    # 4. Bottom loss — estimated bounces
    n_bounces  = max(1, int(r / max(D_water, 1.0)))
    TL_bot_per = bottom_reflection_loss(f_Hz, grazing, bottom_type)
    TL_bot     = n_bounces * TL_bot_per

    # 5. Volume scattering
    TL_vol = volume_scattering_loss(f_Hz, r, D_water)

    # 6. Bubble attenuation (wind > 8 m/s)
    TL_bubble = (bubble_attenuation(f_Hz, min(src_depth, 5.0)) * r
                 if wind_ms > 8.0 else 0.0)

    # 7. Multipath fading (statistical penalty)
    TL_multi = 3.0 if r > D_water else 0.0

    total = TL_geo + TL_abs + 0.3 * TL_surf + 0.5 * TL_bot + TL_vol + TL_bubble

    return {
        'geometric_spherical': TL_geo_sph,
        'geometric_cylindrical': TL_geo_cyl,
        'geometric_practical': TL_geo,
        'absorption_dBm': alpha_FG,
        'absorption_dBkm': alpha_FG * 1000.0,
        'absorption_total': TL_abs,
        'surface_scatter': TL_surf,
        'bottom_per_bounce': TL_bot_per,
        'bottom_bounces': n_bounces,
        'bottom_total': TL_bot,
        'volume_scatter': TL_vol,
        'bubble_attenuation': TL_bubble,
        'multipath': TL_multi,
        'total': total,
        'grazing_angle': grazing,
    }


# ══════════════════════════════════════════════════════════════════
#  BELLHOP ENVIRONMENT PROFILES
# ══════════════════════════════════════════════════════════════════

def get_shallow_water_params(freq=25000):
    """Bellhop parameters for shallow fresh-water pond (NIOT pond test)."""
    depth_max = 3.0
    depths, speeds = ssp_shallow_pond(depth_max, T_surface=28.0)
    return {
        'title':       'NIOT-Shallow-Pond-Test',
        'freq':        float(freq),
        'top_opt':     'SVW',        # S=custom SSP, V=vacuum surface, W=write SHD
        'sigma_top':   0.0,          # smooth surface
        'depth':       depth_max,
        'ssp_depths':  depths,
        'ssp_speeds':  speeds,
        'bot_opt':     'A',          # acousto-elastic half-space
        'sigma_bot':   0.02,         # bottom roughness σ (m)
        'bottom': dict(cp=1520, cs=50, rho=1.5, ap=1.5, as_=1.0),
        'source_depths': [0.5],
        'rec_depths':  np.linspace(0.1, 2.9, 30),
        'n_ranges':    500,
        'r_min_km':    0.001,
        'r_max_km':    0.10,
        'run_type':    'C',          # coherent TL → .shd
        'n_beams':     0,
        'angle_min':   -80.0,
        'angle_max':    80.0,
        # meta
        'env_type':    'shallow',
        'description': 'Pond (~3 m, fresh water, mud bottom, Chennai)',
        'T': 28.0, 'S': 0.0,
        'bottom_type': 'mud',
        'wind_ms': 1.0, 'sea_state': 0,
    }

def get_deep_water_params(freq=25000):
    """Bellhop parameters for coastal/beach deep water (Bay of Bengal)."""
    depth_max = 80.0
    depths, speeds = ssp_coastal_deep(depth_max, T_surface=28.0, S=35.0)
    return {
        'title':       'NIOT-Coastal-BayOfBengal',
        'freq':        float(freq),
        'top_opt':     'SVW',
        'sigma_top':   0.05,         # wave roughness
        'depth':       depth_max,
        'ssp_depths':  depths,
        'ssp_speeds':  speeds,
        'bot_opt':     'A',
        'sigma_bot':   0.05,
        'bottom': dict(cp=1650, cs=200, rho=1.9, ap=0.8, as_=2.5),
        'source_depths': [5.0],
        'rec_depths':  np.linspace(1.0, 75.0, 50),
        'n_ranges':    500,
        'r_min_km':    0.01,
        'r_max_km':    2.00,
        'run_type':    'C',
        'n_beams':     0,
        'angle_min':   -80.0,
        'angle_max':    80.0,
        'env_type':    'deep',
        'description': 'Coastal (~80 m, salt water, sandy bottom, Bay of Bengal)',
        'T': 28.0, 'S': 35.0,
        'bottom_type': 'sand',
        'wind_ms': 5.0, 'sea_state': 2,
    }


# ══════════════════════════════════════════════════════════════════
#  BELLHOP .env FILE GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_env_file(filepath, p, run_type=None):
    """
    Write a valid Bellhop .env file from a parameters dict.
    run_type overrides p['run_type'] (use to run R/A/C separately).
    """
    rtype = run_type or p['run_type']
    depths = p['ssp_depths']
    speeds = p['ssp_speeds']
    bot    = p['bottom']

    lines = []
    lines.append(f"'{p['title']}'")
    lines.append(f"{p['freq']:.1f}")
    lines.append("1")
    lines.append(f"'{p['top_opt']}'")
    lines.append(f"{len(depths)}  {p['sigma_top']:.4f}  {p['depth']:.4f}")
    for d, c in zip(depths, speeds):
        lines.append(f"  {d:.4f}  {c:.4f}  /")
    lines.append(f"'{p['bot_opt']}'  {p['sigma_bot']:.4f}")
    lines.append(f"  {p['depth']:.4f}  {bot['cp']:.1f}  {bot['cs']:.1f}  "
                 f"{bot['rho']:.2f}  {bot['ap']:.2f}  {bot['as_']:.2f}  /")
    # Source depths
    sd = p['source_depths']
    lines.append(f"{len(sd)}")
    lines.append("  " + "  ".join(f"{s:.4f}" for s in sd) + "  /")
    # Receiver depths
    rd = p['rec_depths']
    lines.append(f"{len(rd)}")
    lines.append(f"  {float(rd[0]):.4f}  {float(rd[-1]):.4f}  /")
    # Receiver ranges (km)
    lines.append(f"{p['n_ranges']}")
    lines.append(f"  {p['r_min_km']:.6f}  {p['r_max_km']:.6f}  /")
    # Run
    lines.append(f"'{rtype}'")
    lines.append(f"{p['n_beams']}")
    lines.append(f"  {p['angle_min']:.1f}  {p['angle_max']:.1f}  /")
    z_box = p['depth'] + 10.0
    r_box = p['r_max_km'] * 1.05
    lines.append(f"  0.0  {z_box:.2f}  {r_box:.6f}")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ══════════════════════════════════════════════════════════════════
#  BELLHOP RUNNER — runs 3× for .ray / .shd / .arr
# ══════════════════════════════════════════════════════════════════

def find_bellhop():
    """Search for Bellhop executable across common locations."""
    candidates = ['bellhop', 'bellhop.exe',
                  '/usr/local/bin/bellhop', '/usr/bin/bellhop',
                  os.path.expanduser('~/bellhop/bellhop'),
                  os.path.expanduser('~/bin/bellhop')]
    for cmd in candidates:
        if shutil.which(cmd):
            return cmd
    return None

def run_bellhop_once(work_dir, base_name, bellhop_cmd):
    """Run Bellhop once, return (success, stdout+stderr)."""
    try:
        result = subprocess.run(
            [bellhop_cmd, base_name],
            cwd=work_dir, capture_output=True, text=True, timeout=60
        )
        return True, result.stdout + result.stderr
    except FileNotFoundError:
        return False, f"Bellhop executable not found: {bellhop_cmd}"
    except subprocess.TimeoutExpired:
        return False, "Bellhop timed out (>60 s)"
    except Exception as e:
        return False, str(e)

def run_bellhop_all(work_dir, p, log_fn):
    """
    Run Bellhop three times:
      Pass 1: RunType 'R'  → .ray (ray diagram)
      Pass 2: RunType 'C'  → .shd (coherent TL)
      Pass 3: RunType 'A'  → .arr (eigenray arrivals)
    Returns (success, combined_log_text).
    """
    bellhop_cmd = find_bellhop()
    if bellhop_cmd is None:
        return False, ("Bellhop executable not found in PATH.\n"
                       "Install Bellhop (AT/OALIB) and ensure it is on PATH.\n"
                       "Falling back to Python TL calculator.")

    base = 'niot_run'
    env_path = os.path.join(work_dir, base + '.env')
    full_log = ""

    for rtype, label in [('R', 'Ray trace'), ('C', 'Coherent TL'), ('A', 'Arrivals')]:
        generate_env_file(env_path, p, run_type=rtype)
        log_fn(f"[BELLHOP] Running {label} ({rtype})...")
        ok, out = run_bellhop_once(work_dir, base, bellhop_cmd)
        full_log += f"\n{'='*40}\nPass: {label} (RunType={rtype})\n{'='*40}\n{out}\n"

        # Rename outputs so each pass keeps its files
        for ext in ['.prt', '.ray', '.shd', '.arr']:
            src = os.path.join(work_dir, base + ext)
            if os.path.exists(src):
                suffix = {'R': '_ray', 'C': '_shd', 'A': '_arr'}[rtype]
                dst = os.path.join(work_dir, f'niot{suffix}' + ext)
                shutil.copy(src, dst)

        if not ok:
            log_fn(f"[BELLHOP] ✗ {label} failed: {out[:200]}")

    # Also keep a canonical .env for display
    generate_env_file(env_path, p)
    return True, full_log


# ══════════════════════════════════════════════════════════════════
#  BELLHOP FILE READERS
# ══════════════════════════════════════════════════════════════════

def read_text_file(path):
    try:
        with open(path, 'r', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"[Read error] {e}"

def read_shd_file(path):
    """
    Read Bellhop .shd binary (FORTRAN unformatted sequential).
    Returns (ranges_km, depths_m, TL_dB 2-D array, title_str)
    or (None, None, None, error_msg).
    """
    def _rec(f):
        raw = f.read(4)
        if len(raw) < 4:
            raise EOFError
        n = struct.unpack('i', raw)[0]
        data = f.read(abs(n))
        f.read(4)           # trailing Fortran record marker
        return data

    try:
        with open(path, 'rb') as f:
            _rec(f)                                     # recl
            title = _rec(f).decode('utf-8', errors='replace').strip()
            _rec(f)                                     # PlotType
            rec4  = _rec(f)
            # freq0, theta, Nsd, Nrd, Nrr, atten
            freq0  = struct.unpack('f', rec4[0:4])[0]
            Nsd    = struct.unpack('i', rec4[8:12])[0]
            Nrd    = struct.unpack('i', rec4[12:16])[0]
            Nrr    = struct.unpack('i', rec4[16:20])[0]
            _rec(f)                                     # freqVec
            _rec(f)                                     # thetaVec
            sd_raw = _rec(f)
            rd_raw = _rec(f)
            rr_raw = _rec(f)

            Nsd = max(Nsd, 1)
            Nrd = max(Nrd, 1)
            Nrr = max(Nrr, 1)

            rds = np.frombuffer(rd_raw, dtype=np.float32)[:Nrd]
            rrs = np.frombuffer(rr_raw, dtype=np.float32)[:Nrr]

            pressure = np.zeros((Nrd, Nrr), dtype=complex)
            for isd in range(Nsd):
                for ird in range(Nrd):
                    try:
                        rec = _rec(f)
                        n_cmp = min(Nrr, len(rec) // 8)
                        p_row = np.frombuffer(rec[:n_cmp*8], dtype=np.complex64)
                        pressure[ird, :n_cmp] = p_row
                    except EOFError:
                        break

        TL = -20.0 * np.log10(np.abs(pressure) + 1e-30)
        return rrs, rds, TL, title
    except Exception as e:
        return None, None, None, f"SHD read error: {e}"

def parse_ray_file(content):
    """
    Parse Bellhop .ray ASCII file.
    Returns list of (r_km_array, z_m_array) tuples per ray.
    """
    rays = []
    lines = content.strip().splitlines()
    i = 0
    # Skip header lines (title, freq, Nsd, SD, NBeams)
    while i < len(lines):
        line = lines[i].strip()
        # Each ray block starts with: alpha  nsteps  NumTopBnc  NumBotBnc
        try:
            parts = line.split()
            if len(parts) == 4:
                try:
                    float(parts[0])
                    npts = int(parts[1])
                    r_pts, z_pts = [], []
                    for j in range(1, npts + 1):
                        if i + j < len(lines):
                            xy = lines[i + j].split()
                            if len(xy) >= 2:
                                r_pts.append(float(xy[0]))
                                z_pts.append(float(xy[1]))
                    if len(r_pts) > 1:
                        rays.append((np.array(r_pts), np.array(z_pts)))
                    i += npts + 1
                    continue
                except ValueError:
                    pass
        except Exception:
            pass
        i += 1
    return rays


# ══════════════════════════════════════════════════════════════════
#  PYTHON FALLBACK TL (when Bellhop is absent)
# ══════════════════════════════════════════════════════════════════

def compute_tl_python(params):
    """
    Ray-theory approximation TL over a 2-D (range × depth) grid.
    Returns (ranges_km, depths_m, TL_dB 2-D array).
    """
    ranges_m = np.linspace(params['r_min_km'] * 1000,
                            params['r_max_km'] * 1000, 200)
    depths   = params['rec_depths']
    TL = np.zeros((len(depths), len(ranges_m)))

    for i, rd in enumerate(depths):
        for j, r in enumerate(ranges_m):
            loss = total_transmission_loss(
                r, params['freq'], params['T'], params['S'],
                params['depth'], params['source_depths'][0],
                float(rd), params['bottom_type'],
                params['wind_ms'], params['sea_state']
            )
            TL[i, j] = loss['total']
    return ranges_m / 1000.0, np.array(depths, dtype=float), TL


# ══════════════════════════════════════════════════════════════════
#  MAIN GUI APPLICATION
# ══════════════════════════════════════════════════════════════════

class USBLBellhopApp:

    # ── USBL / MAVLink config ────────────────────────────────────
    USBL_IP   = "192.168.0.187"
    USBL_PORT = 9200
    MAVLINK_UDP  = "udp:127.0.0.1:14590"
    MODEM_PORTS  = [9001, 9002]
    MODEM_IP     = "127.0.0.1"
    MAX_PAYLOAD  = 128
    COMPRESS_LVL = 6
    ALLOWED_MSGS = [
        "COMMAND_LONG","MISSION_ITEM","MISSION_COUNT",
        "MISSION_CURRENT","SET_MODE","HEARTBEAT","GLOBAL_POSITION_INT"
    ]
    MODE_MAP = {
        0:"STABILIZE",1:"ACRO",2:"ALT_HOLD",3:"AUTO",4:"GUIDED",
        5:"LOITER",6:"RTL",7:"CIRCLE",9:"LAND",11:"DRIFT",
        13:"SPORT",16:"POSHOLD",17:"BRAKE",20:"GUIDED_NOGPS",21:"SMART_RTL"
    }

    def __init__(self, root):
        self.root = root
        self.root.title("NIOT — USBL + Bellhop Acoustic System")
        self.root.geometry("1450x900")
        self.root.configure(bg="#060612")

        self.log_queue   = Queue()
        self.is_shallow  = True
        self.freq_var    = tk.StringVar(value="25000")
        self.work_dir    = tempfile.mkdtemp(prefix="bellhop_niot_")
        self.params      = get_shallow_water_params()
        self.msg_counter = 0
        self.tx_running  = False
        self.rx_running  = False
        self.rx_counts   = {9001: 0, 9002: 0}
        self.bellhop_results = {}   # store latest simulation outputs

        self._build_gui()
        self._poll_logs()

    # ────────────────────────────────────────────────────────────
    #  GUI CONSTRUCTION
    # ────────────────────────────────────────────────────────────

    def _build_gui(self):
        # ── Header bar ──────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#060612", height=56)
        hdr.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(hdr, text="⚓  NIOT · USBL + BELLHOP ACOUSTIC SYSTEM",
                 bg="#060612", fg="#00ffc8",
                 font=("Courier New", 17, "bold")).pack(side="left", padx=10)

        # Frequency
        ff = tk.Frame(hdr, bg="#060612")
        ff.pack(side="right", padx=14)
        tk.Label(ff, text="Freq (Hz):", bg="#060612", fg="#88aaaa",
                 font=("Courier New", 10)).pack(side="left")
        tk.Entry(ff, textvariable=self.freq_var, width=8,
                 bg="#111128", fg="#ffffff",
                 font=("Courier New", 10),
                 insertbackground="white").pack(side="left", padx=4)

        # ── WATER MODE TOGGLE BUTTON ────────────────────────────
        self.toggle_btn = tk.Button(
            hdr,
            text="🏊  SHALLOW WATER\n        (Pond Mode)",
            bg="#0066cc", fg="#ffffff",
            font=("Courier New", 11, "bold"),
            width=22, height=2, relief="ridge", bd=3,
            activebackground="#0055aa",
            command=self._toggle_water_mode
        )
        self.toggle_btn.pack(side="right", padx=8)

        self.mode_indicator = tk.Label(
            hdr, text="▶ SHALLOW", bg="#060612",
            fg="#33ccff", font=("Courier New", 12, "bold"))
        self.mode_indicator.pack(side="right", padx=4)

        # ── Notebook ────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TNotebook',       background='#060612', borderwidth=0)
        style.configure('Dark.TNotebook.Tab',   background='#111128',
                        foreground='#00ffc8', padding=[12, 5], font=('Courier New', 9, 'bold'))
        style.map('Dark.TNotebook.Tab', background=[('selected','#0066cc')])

        self.nb = ttk.Notebook(self.root, style='Dark.TNotebook')
        self.nb.pack(fill="both", expand=True, padx=8, pady=6)

        tabs_def = [
            ('tab_usbl',   "📡  USBL COMMS"),
            ('tab_bell',   "🔊  BELLHOP SIM"),
            ('tab_tl',     "📊  TRANS. LOSS"),
            ('tab_rays',   "〰  RAY DIAGRAM"),
            ('tab_losses', "📉  LOSS ANALYSIS"),
            ('tab_ssp',    "🌊  SOUND SPEED"),
            ('tab_files',  "📁  FILE VIEWER"),
        ]
        for attr, label in tabs_def:
            frame = tk.Frame(self.nb, bg="#070714")
            setattr(self, attr, frame)
            self.nb.add(frame, text=label)

        self._build_usbl_tab()
        self._build_bellhop_tab()
        self._build_tl_tab()
        self._build_ray_tab()
        self._build_losses_tab()
        self._build_ssp_tab()
        self._build_files_tab()

    # ── Tab: USBL COMMS ─────────────────────────────────────────

    def _build_usbl_tab(self):
        tab = self.tab_usbl

        # Status panels row
        row = tk.Frame(tab, bg="#070714")
        row.pack(fill="x", padx=6, pady=5)

        def _panel(parent, title):
            f = tk.LabelFrame(parent, text=title, bg="#0d0d25",
                              fg="#00ffc8", bd=2, relief="groove",
                              font=("Courier New", 9, "bold"))
            f.pack(side="left", expand=True, fill="both", padx=4)
            return f

        # TX
        tx_panel = _panel(row, "TX STATUS")
        self.txv_mode  = tk.StringVar(value="Mode: ---")
        self.txv_pos   = tk.StringVar(value="Position: ---")
        self.txv_count = tk.StringVar(value="Packets TX: 0")
        self.txv_usbl  = tk.StringVar(value="USBL: UNKNOWN")
        for v in [self.txv_mode, self.txv_pos, self.txv_count, self.txv_usbl]:
            tk.Label(tx_panel, textvariable=v, bg="#0d0d25",
                    fg="#ddffdd", font=("Courier New", 9)).pack(anchor="w", padx=4, pady=1)

        # RX
        rx_panel = _panel(row, "RX STATUS")
        self.rxv_last  = tk.StringVar(value="Last msg: ---")
        self.rxv_count = tk.StringVar(value="Received: 0")
        for v in [self.rxv_last, self.rxv_count]:
            tk.Label(rx_panel, textvariable=v, bg="#0d0d25",
                    fg="#ddffdd", font=("Courier New", 9)).pack(anchor="w", padx=4, pady=1)

        # Modems
        mod_panel = _panel(row, "MODEMS")
        self.modv1 = tk.StringVar(value="Port 9001: ○ OFF")
        self.modv2 = tk.StringVar(value="Port 9002: ○ OFF")
        for v in [self.modv1, self.modv2]:
            tk.Label(mod_panel, textvariable=v, bg="#0d0d25",
                    fg="#ffcc44", font=("Courier New", 9)).pack(anchor="w", padx=4, pady=1)

        # Acoustic Channel (live)
        ac_panel = _panel(row, "ACOUSTIC CHANNEL")
        self.acv_tl   = tk.StringVar(value="TL: --- dB")
        self.acv_snr  = tk.StringVar(value="SNR: --- dB")
        self.acv_range= tk.StringVar(value="Range: --- m")
        self.acv_alpha= tk.StringVar(value="α: --- dB/km")
        for v in [self.acv_tl, self.acv_snr, self.acv_range, self.acv_alpha]:
            tk.Label(ac_panel, textvariable=v, bg="#0d0d25",
                    fg="#ddffdd", font=("Courier New", 9)).pack(anchor="w", padx=4, pady=1)

        # Buttons
        btn_row = tk.Frame(tab, bg="#070714")
        btn_row.pack(pady=6)

        def _btn(parent, text, bg, cmd):
            b = tk.Button(parent, text=text, bg=bg, fg="white",
                         font=("Courier New", 10, "bold"),
                         padx=10, pady=4, relief="raised", bd=2,
                         activebackground=bg, command=cmd)
            b.pack(side="left", padx=6)
            return b

        self.btn_tx = _btn(btn_row, "▶ Start TX",  "#006633", self._start_tx)
        self.btn_rx = _btn(btn_row, "▶ Start RX",  "#003399", self._start_rx)
        _btn(btn_row, "⟳ Acoustic Update", "#664400", self._update_acoustic_live)

        # Log
        tk.Label(tab, text="SYSTEM LOG", bg="#070714",
                fg="#556677", font=("Courier New", 8)).pack(anchor="w", padx=10)
        self.log_box = scrolledtext.ScrolledText(
            tab, bg="#040410", fg="#00ffc8",
            font=("Courier New", 9), insertbackground="white")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Tab: BELLHOP SIM ─────────────────────────────────────────

    def _build_bellhop_tab(self):
        tab = self.tab_bell

        left = tk.Frame(tab, bg="#070714", width=340)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="ENVIRONMENT PARAMETERS",
                 bg="#070714", fg="#00ffc8",
                 font=("Courier New", 10, "bold")).pack(pady=4)

        self.params_txt = scrolledtext.ScrolledText(
            left, bg="#040410", fg="#aaffcc",
            font=("Courier New", 8), width=40)
        self.params_txt.pack(fill="both", expand=True, padx=3)

        right = tk.Frame(tab, bg="#070714")
        right.pack(side="left", fill="both", expand=True, padx=5)

        # Control buttons
        ctrl = tk.Frame(right, bg="#070714")
        ctrl.pack(fill="x", pady=5)

        def _cbtn(parent, text, bg, cmd, w=18):
            return tk.Button(parent, text=text, bg=bg, fg="white",
                            font=("Courier New", 10, "bold"),
                            width=w, height=2, relief="raised", bd=2,
                            command=cmd).pack(side="left", padx=4)

        _cbtn(ctrl, "▶ RUN BELLHOP",    "#006644", self._run_bellhop)
        _cbtn(ctrl, "📊 Python TL",     "#004499", self._run_python_tl, 14)
        _cbtn(ctrl, "📋 Preview .env",  "#443300", self._preview_env,   14)
        _cbtn(ctrl, "📂 Open Dir",      "#330044", self._open_work_dir, 12)

        self.bell_status = tk.StringVar(value="Status: Ready — Press RUN BELLHOP")
        tk.Label(right, textvariable=self.bell_status,
                bg="#070714", fg="#ffcc44",
                font=("Courier New", 10)).pack(anchor="w", padx=6, pady=2)

        tk.Label(right, text="BELLHOP CONSOLE OUTPUT:",
                bg="#070714", fg="#556677",
                font=("Courier New", 8)).pack(anchor="w", padx=6)

        self.bell_log = scrolledtext.ScrolledText(
            right, bg="#040410", fg="#ffcc44",
            font=("Courier New", 8))
        self.bell_log.pack(fill="both", expand=True, padx=5, pady=4)

        self._refresh_params_display()

    # ── Tab: TRANSMISSION LOSS ────────────────────────────────────

    def _build_tl_tab(self):
        tab = self.tab_tl
        self.fig_tl  = Figure(figsize=(13, 6), facecolor="#040410")
        self.ax_tl   = self.fig_tl.add_subplot(111)
        self._style_ax(self.ax_tl, "Transmission Loss — run simulation first",
                       "Range (km)", "Depth (m)")
        self.canvas_tl = FigureCanvasTkAgg(self.fig_tl, tab)
        self.canvas_tl.get_tk_widget().pack(fill="both", expand=True)

        self.tl_info = tk.StringVar(value="Run Bellhop or Python TL to populate this view")
        tk.Label(tab, textvariable=self.tl_info,
                bg="#070714", fg="#888888",
                font=("Courier New", 8)).pack()

    # ── Tab: RAY DIAGRAM ─────────────────────────────────────────

    def _build_ray_tab(self):
        tab = self.tab_rays
        self.fig_ray  = Figure(figsize=(13, 6), facecolor="#040410")
        self.ax_ray   = self.fig_ray.add_subplot(111)
        self._style_ax(self.ax_ray, "Ray Diagram — run Bellhop first",
                       "Range (km)", "Depth (m)")
        self.canvas_ray = FigureCanvasTkAgg(self.fig_ray, tab)
        self.canvas_ray.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab: LOSS ANALYSIS ────────────────────────────────────────

    def _build_losses_tab(self):
        tab = self.tab_losses

        left = tk.Frame(tab, bg="#070714", width=300)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="INPUT PARAMETERS", bg="#070714",
                fg="#00ffc8", font=("Courier New", 9, "bold")).pack(pady=4)

        self._loss_vars = {}
        entries = [
            ("Range (m):",       "range",    "100"),
            ("Frequency (Hz):",  "freq",     "25000"),
            ("Temperature (°C):","temp",     "28"),
            ("Salinity (ppt):",  "sal",      "35"),
            ("Water Depth (m):", "depth",    "80"),
            ("Wind Speed (m/s):","wind",     "5"),
            ("Sea State:",       "sea",      "2"),
            ("Source Depth (m):","srcdepth", "5"),
        ]
        for label, key, default in entries:
            row = tk.Frame(left, bg="#070714")
            row.pack(fill="x", padx=4, pady=2)
            tk.Label(row, text=label, bg="#070714", fg="#aaaaaa",
                    font=("Courier New", 8), width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self._loss_vars[key] = var
            tk.Entry(row, textvariable=var, width=9,
                    bg="#111128", fg="white",
                    font=("Courier New", 8)).pack(side="left")

        # Bottom type
        row = tk.Frame(left, bg="#070714")
        row.pack(fill="x", padx=4, pady=2)
        tk.Label(row, text="Bottom Type:", bg="#070714", fg="#aaaaaa",
                font=("Courier New", 8), width=20, anchor="w").pack(side="left")
        self._loss_bottom = tk.StringVar(value="sand")
        ttk.Combobox(row, textvariable=self._loss_bottom,
                    values=["sand","mud","clay","rock","gravel","silt"],
                    width=10, font=("Courier New", 8)).pack(side="left")

        tk.Button(left, text="▶ Calculate ALL Losses",
                 bg="#006644", fg="white",
                 font=("Courier New", 9, "bold"),
                 command=self._calc_losses).pack(pady=8, fill="x", padx=4)

        self.loss_out = scrolledtext.ScrolledText(
            left, bg="#040410", fg="#ffcc44",
            font=("Courier New", 8))
        self.loss_out.pack(fill="both", expand=True, padx=3)

        # Chart area
        right = tk.Frame(tab, bg="#070714")
        right.pack(side="left", fill="both", expand=True)
        self.fig_loss = Figure(figsize=(9, 7), facecolor="#040410")
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, right)
        self.canvas_loss.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab: SOUND SPEED ─────────────────────────────────────────

    def _build_ssp_tab(self):
        tab = self.tab_ssp
        self.fig_ssp = Figure(figsize=(13, 6), facecolor="#040410")
        gs = self.fig_ssp.add_gridspec(1, 2, wspace=0.35)
        self.ax_ssp_s = self.fig_ssp.add_subplot(gs[0])
        self.ax_ssp_d = self.fig_ssp.add_subplot(gs[1])
        self.canvas_ssp = FigureCanvasTkAgg(self.fig_ssp, tab)
        self.canvas_ssp.get_tk_widget().pack(fill="both", expand=True)
        self._plot_ssp()

    # ── Tab: FILE VIEWER ─────────────────────────────────────────

    def _build_files_tab(self):
        tab = self.tab_files

        top = tk.Frame(tab, bg="#070714")
        top.pack(fill="x", padx=6, pady=4)

        tk.Label(top, text="Bellhop Output Files:",
                bg="#070714", fg="#00ffc8",
                font=("Courier New", 10, "bold")).pack(side="left", padx=4)

        for label, ext in [('.env','_run.env'), ('.prt','_ray.prt'),
                            ('.ray','_ray.ray'), ('.arr','_arr.arr'),
                            ('.shd info','_shd.shd')]:
            tk.Button(top, text=label, bg="#111128", fg="#00ffc8",
                     font=("Courier New", 9), relief="groove",
                     command=lambda e=ext: self._show_file(e)
                     ).pack(side="left", padx=2)

        tk.Button(top, text="List All Files", bg="#220022", fg="#cc88ff",
                 font=("Courier New", 9),
                 command=self._list_work_dir).pack(side="right", padx=6)

        self.wd_label = tk.Label(tab,
            text=f"Work dir: {self.work_dir}",
            bg="#070714", fg="#444466",
            font=("Courier New", 7))
        self.wd_label.pack(anchor="w", padx=8)

        self.file_view = scrolledtext.ScrolledText(
            tab, bg="#030310", fg="#ccffcc",
            font=("Courier New", 8))
        self.file_view.pack(fill="both", expand=True, padx=6, pady=4)

    # ────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────

    def _style_ax(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#040410")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#333355")

    def log(self, msg):
        self.log_queue.put(msg)

    def _poll_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            ts  = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_box.see(tk.END)
        self.root.after(100, self._poll_logs)

    # ────────────────────────────────────────────────────────────
    #  WATER MODE TOGGLE
    # ────────────────────────────────────────────────────────────

    def _toggle_water_mode(self):
        freq = int(self.freq_var.get() or 25000)
        if self.is_shallow:
            self.is_shallow = False
            self.params     = get_deep_water_params(freq)
            self.toggle_btn.config(
                text="🌊  DEEP WATER\n    (Coastal Mode)",
                bg="#cc4400")
            self.mode_indicator.config(text="▶ DEEP", fg="#ff8833")
            self.log("[MODE] ── Switched to DEEP WATER (Coastal / Bay of Bengal) ──")
        else:
            self.is_shallow = True
            self.params     = get_shallow_water_params(freq)
            self.toggle_btn.config(
                text="🏊  SHALLOW WATER\n        (Pond Mode)",
                bg="#0066cc")
            self.mode_indicator.config(text="▶ SHALLOW", fg="#33ccff")
            self.log("[MODE] ── Switched to SHALLOW WATER (Pond Test) ──")

        self._refresh_params_display()
        self._plot_ssp()

    def _refresh_params_display(self):
        p = self.params
        ds, sp = p['ssp_depths'], p['ssp_speeds']
        txt = (f"{'='*38}\n"
               f"MODE: {'SHALLOW WATER' if p['env_type']=='shallow' else 'DEEP WATER'}\n"
               f"{p['description']}\n"
               f"{'='*38}\n\n"
               f"FREQUENCY     : {p['freq']:.0f} Hz\n"
               f"WATER DEPTH   : {p['depth']:.1f} m\n"
               f"TEMPERATURE   : {p['T']:.1f} °C\n"
               f"SALINITY      : {p['S']:.1f} ppt\n\n"
               f"SOUND SPEED   : {min(sp):.2f} – {max(sp):.2f} m/s\n\n"
               f"SOURCE DEPTH  : {p['source_depths'][0]:.2f} m\n"
               f"REC DEPTHS    : {float(p['rec_depths'][0]):.1f} – {float(p['rec_depths'][-1]):.1f} m\n"
               f"RANGE         : {p['r_min_km']*1000:.0f} – {p['r_max_km']*1000:.0f} m\n\n"
               f"BOTTOM TYPE   : {p['bottom_type'].upper()}\n"
               f"  cp  = {p['bottom']['cp']} m/s\n"
               f"  cs  = {p['bottom']['cs']} m/s\n"
               f"  ρ   = {p['bottom']['rho']} g/cc\n"
               f"  αp  = {p['bottom']['ap']} dB/λ\n"
               f"  αs  = {p['bottom']['as_']} dB/λ\n"
               f"  σ   = {p['sigma_bot']} m (roughness)\n\n"
               f"SURFACE:\n"
               f"  Wind : {p['wind_ms']} m/s\n"
               f"  Sea  : {p['sea_state']} Beaufort\n\n"
               f"BELLHOP:\n"
               f"  Runs : R → C → A\n"
               f"  Beams: {p['angle_min']}° → {p['angle_max']}°\n"
               f"  N_r  : {p['n_ranges']}\n"
               f"{'='*38}")
        self.params_txt.delete("1.0", tk.END)
        self.params_txt.insert(tk.END, txt)

    # ────────────────────────────────────────────────────────────
    #  BELLHOP ACTIONS
    # ────────────────────────────────────────────────────────────

    def _run_bellhop(self):
        self.bell_status.set("Status: Running Bellhop (3 passes)…")
        threading.Thread(target=self._bellhop_thread, daemon=True).start()

    def _bellhop_thread(self):
        freq = int(self.freq_var.get() or 25000)
        params = (get_shallow_water_params(freq) if self.is_shallow
                  else get_deep_water_params(freq))
        self.params = params

        self.log("[BELLHOP] Generating environment file…")
        ok, full_log = run_bellhop_all(self.work_dir, params, self.log)

        self.bell_log.delete("1.0", tk.END)
        self.bell_log.insert(tk.END, full_log)

        if ok:
            self.bell_status.set("Status: ✓ Bellhop complete — loading results")
            self.log("[BELLHOP] ✓ All passes done")
            self.root.after(50, lambda: self._load_results(params))
        else:
            self.bell_status.set("Status: Bellhop unavailable → Python fallback")
            self.log("[BELLHOP] Bellhop not installed — using Python TL")
            self.root.after(50, self._run_python_tl)

    def _load_results(self, params):
        """Read and display all Bellhop output files."""
        wdir = self.work_dir

        # RAY file
        ray_path = os.path.join(wdir, 'niot_ray.ray')
        if os.path.exists(ray_path):
            content = read_text_file(ray_path)
            rays    = parse_ray_file(content)
            self.bellhop_results['rays'] = rays
            self.log(f"[FILE] .ray loaded — {len(rays)} rays")
            self.root.after(0, lambda: self._plot_rays(rays, params))

        # SHD file
        shd_path = os.path.join(wdir, 'niot_shd.shd')
        if os.path.exists(shd_path):
            rrs, rds, TL, title = read_shd_file(shd_path)
            if TL is not None:
                self.bellhop_results['shd'] = (rrs, rds, TL)
                self.log(f"[FILE] .shd loaded — TL grid {TL.shape}")
                self.root.after(0, lambda: self._plot_tl(rrs, rds, TL,
                    f"TL — {params['description']}"))
            else:
                self.log(f"[FILE] .shd parse error: {title}")

        # ARR file
        arr_path = os.path.join(wdir, 'niot_arr.arr')
        if os.path.exists(arr_path):
            self.log("[FILE] .arr loaded")

        # List all generated files
        all_files = sorted(os.listdir(wdir))
        self.log(f"[FILES] Generated: {', '.join(all_files)}")
        self.bell_status.set(f"Status: ✓ Done — {len(all_files)} files in work dir")

    # ── Python TL fallback ────────────────────────────────────────

    def _run_python_tl(self):
        self.bell_status.set("Status: Computing Python TL…")
        threading.Thread(target=self._python_tl_thread, daemon=True).start()

    def _python_tl_thread(self):
        self.log("[PYTHON] Computing transmission loss grid…")
        try:
            rrs, rds, TL = compute_tl_python(self.params)
            self.log(f"[PYTHON] Done — grid {TL.shape[0]}×{TL.shape[1]}")
            self.root.after(0, lambda: self._plot_tl(rrs, rds, TL,
                f"TL (Python approx.) — {self.params['description']}"))
        except Exception as e:
            self.log(f"[ERROR] Python TL: {e}")

    # ────────────────────────────────────────────────────────────
    #  PLOT FUNCTIONS
    # ────────────────────────────────────────────────────────────

    def _plot_tl(self, ranges_km, depths_m, TL, title):
        ax = self.ax_tl
        ax.clear()
        ax.set_facecolor("#040410")

        R, D = np.meshgrid(ranges_km, depths_m)
        TL_c = np.clip(TL, 20, 110)
        im   = ax.pcolormesh(R, D, TL_c, cmap='jet_r', shading='auto',
                             vmin=30, vmax=100)

        try:
            cb = self.fig_tl.colorbar(im, ax=ax, pad=0.01)
            cb.set_label('TL (dB)', color='white')
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
        except Exception:
            pass

        ax.invert_yaxis()
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("Range (km)", color="white")
        ax.set_ylabel("Depth (m)",  color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#333355")

        self.canvas_tl.draw()
        self.tl_info.set(f"TL: {TL.min():.1f}–{TL.max():.1f} dB | "
                          f"Grid: {len(depths_m)}×{len(ranges_km)}")
        self.nb.select(self.tab_tl)

    def _plot_rays(self, rays, params):
        ax = self.ax_ray
        ax.clear()
        ax.set_facecolor("#040410")

        palette = plt.cm.rainbow(np.linspace(0, 1, max(len(rays), 1)))
        for idx, (r_pts, z_pts) in enumerate(rays):
            ax.plot(r_pts, z_pts, '-',
                   color=palette[idx % len(palette)],
                   linewidth=0.6, alpha=0.75)

        ax.invert_yaxis()
        ax.set_ylim(params['depth'] + 1, -0.5)
        ax.axhline(y=params['depth'], color='#8B5A1A',
                  linewidth=2.5, label='Bottom', zorder=5)
        ax.axhline(y=0.0, color='#1a6688',
                  linewidth=2.5, label='Surface', zorder=5)
        ax.set_xlabel("Range (km)", color="white")
        ax.set_ylabel("Depth (m)",  color="white")
        ax.set_title(f"Ray Diagram — {len(rays)} rays | {params['description']}",
                    color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0d0d25", labelcolor="white", fontsize=8)
        for s in ax.spines.values():
            s.set_color("#333355")

        self.canvas_ray.draw()
        self.nb.select(self.tab_rays)
        self.log(f"[RAY] Plotted {len(rays)} rays")

    def _plot_ssp(self):
        for ax in [self.ax_ssp_s, self.ax_ssp_d]:
            ax.clear()
            ax.set_facecolor("#040410")
            for s in ax.spines.values():
                s.set_color("#333355")

        # Shallow
        ds, ss = ssp_shallow_pond(3.0, 28.0)
        self.ax_ssp_s.plot(ss, ds, color="#33ccff", linewidth=2.5)
        self.ax_ssp_s.fill_betweenx(ds, ss.min()-1, ss, alpha=0.15, color="#33ccff")
        self.ax_ssp_s.invert_yaxis()
        self.ax_ssp_s.set_title("Shallow SSP — Pond (28°C, Fresh)",
                                 color="white", fontsize=9)
        self.ax_ssp_s.set_xlabel("c (m/s)", color="white")
        self.ax_ssp_s.set_ylabel("Depth (m)", color="white")
        self.ax_ssp_s.tick_params(colors="white")
        self.ax_ssp_s.grid(True, alpha=0.15)
        self.ax_ssp_s.axvline(ss.min(), color="yellow", ls="--", alpha=0.6)
        self.ax_ssp_s.text(ss.min()+0.3, ds[-1]*0.6,
                           f"c_min={ss.min():.0f} m/s",
                           color="yellow", fontsize=7)

        # Deep
        dd, sd = ssp_coastal_deep(80.0, 28.0, 35.0)
        self.ax_ssp_d.plot(sd, dd, color="#ff7744", linewidth=2.5)
        self.ax_ssp_d.fill_betweenx(dd, sd.min()-1, sd, alpha=0.15, color="#ff7744")
        self.ax_ssp_d.invert_yaxis()
        self.ax_ssp_d.set_title("Deep SSP — Coastal Bay of Bengal (S=35 ppt)",
                                  color="white", fontsize=9)
        self.ax_ssp_d.set_xlabel("c (m/s)", color="white")
        self.ax_ssp_d.set_ylabel("Depth (m)", color="white")
        self.ax_ssp_d.tick_params(colors="white")
        self.ax_ssp_d.grid(True, alpha=0.15)
        idx_min = np.argmin(sd)
        self.ax_ssp_d.axhline(dd[idx_min], color="yellow", ls="--", alpha=0.6)
        self.ax_ssp_d.text(sd.min()+0.3, dd[idx_min]-2,
                           f"c_min={sd.min():.0f} m/s\n@ {dd[idx_min]:.0f} m",
                           color="yellow", fontsize=7)

        self.fig_ssp.tight_layout()
        self.fig_ssp.patch.set_facecolor("#040410")
        self.canvas_ssp.draw()

    # ────────────────────────────────────────────────────────────
    #  LOSS ANALYSIS
    # ────────────────────────────────────────────────────────────

    def _calc_losses(self):
        try:
            r       = float(self._loss_vars['range'].get())
            f       = float(self._loss_vars['freq'].get())
            T       = float(self._loss_vars['temp'].get())
            S       = float(self._loss_vars['sal'].get())
            D       = float(self._loss_vars['depth'].get())
            wind    = float(self._loss_vars['wind'].get())
            sea     = float(self._loss_vars['sea'].get())
            sd      = float(self._loss_vars['srcdepth'].get())
            bottom  = self._loss_bottom.get()

            loss = total_transmission_loss(r, f, T, S, D, sd, D/2, bottom, wind, sea)

            alpha_FG   = francois_garrison_absorption(f, T, S, D) * 1000  # dB/km
            alpha_Th   = thorp_absorption(f)                               # dB/km
            c_sea      = sound_speed_seawater(T, S, D)
            c_fresh    = sound_speed_freshwater(T, D)
            lam        = c_sea / f
            f_dop      = doppler_shift(f, 1.0, 0.0, c_sea)
            N          = ambient_noise_wenz(f, int(sea))

            BL = {a: bottom_reflection_loss(f, a, bottom)
                  for a in [5, 15, 30, 45, 60, 75]}
            SL_wind = {a: surface_scattering_loss(f, wind, a)
                       for a in [5, 30, 60]}
            vol_loss = volume_scattering_loss(f, r, D)
            bub_loss = bubble_attenuation(f, min(sd, 5.0))

            SL_src  = 180.0
            SNR     = SL_src - loss['total'] - N

            out = f"""
{'═'*44}
NIOT ACOUSTIC LOSS ANALYSIS
{'═'*44}
Frequency : {f/1000:.2f} kHz
Range     : {r} m
Water depth: {D} m
{'─'*44}
SOUND SPEED
  Seawater (T={T}°C, S={S}ppt, D={D}m)
    c = {c_sea:.3f} m/s
  Fresh water (T={T}°C)
    c = {c_fresh:.3f} m/s
  Wavelength λ = {lam*100:.3f} cm

GEOMETRIC SPREADING
  Spherical   20·log(r) = {loss['geometric_spherical']:.2f} dB
  Cylindrical 10·log(r) = {loss['geometric_cylindrical']:.2f} dB
  Practical (mixed)     = {loss['geometric_practical']:.2f} dB

ABSORPTION COEFFICIENTS
  Francois-Garrison : {alpha_FG:.5f} dB/km
  Thorp (approx)    : {alpha_Th:.5f} dB/km
  Total at {r} m   : {loss['absorption_total']:.4f} dB

BOTTOM REFLECTION LOSS — {bottom.upper()}
  At  5° grazing : {BL[5]:.2f} dB
  At 15° grazing : {BL[15]:.2f} dB
  At 30° grazing : {BL[30]:.2f} dB
  At 45° grazing : {BL[45]:.2f} dB
  At 60° grazing : {BL[60]:.2f} dB
  At 75° grazing : {BL[75]:.2f} dB
  Estimated bounces: {loss['bottom_bounces']}
  Effective BL   : {loss['bottom_total']:.2f} dB

SURFACE SCATTERING LOSS (wind={wind} m/s)
  At  5° grazing : {SL_wind[5]:.2f} dB
  At 30° grazing : {SL_wind[30]:.2f} dB
  At 60° grazing : {SL_wind[60]:.2f} dB

VOLUME SCATTERING
  DSL (biol. scatter): {vol_loss:.4f} dB

BUBBLE ATTENUATION
  Near-surface extra : {bub_loss*r:.4f} dB/m × {r} m
  (significant only wind > 8 m/s)

AMBIENT NOISE (Wenz)
  Sea State {int(sea)} : {N:.2f} dB re 1 µPa²/Hz

DOPPLER SHIFT
  For v_src = 1 m/s:
    Δf = {f_dop - f:.3f} Hz  ({(f_dop/f-1)*1e6:.2f} ppm)

TL BUDGET SUMMARY
  Geometric (practical) : {loss['geometric_practical']:.2f} dB
  Absorption            : {loss['absorption_total']:.4f} dB
  Surface scatter (×0.3): {0.3*loss['surface_scatter']:.2f} dB
  Bottom refl  (×0.5)   : {0.5*loss['bottom_total']:.2f} dB
  Volume scatter         : {loss['volume_scatter']:.4f} dB
  Bubble atten           : {loss['bubble_attenuation']:.4f} dB
  ─────────────────────────────────────
  TOTAL TL               : {loss['total']:.2f} dB

LINK BUDGET (Source Level = 180 dB re µPa)
  SL  =  180.00 dB
  TL  = -{loss['total']:.2f} dB
  NL  = -{N:.2f} dB
  ───────────────
  SNR ≈ {SNR:.2f} dB
{'═'*44}
"""
            self.loss_out.delete("1.0", tk.END)
            self.loss_out.insert(tk.END, out)
            self._plot_loss_curves(f, T, S, D, bottom, wind, sea)

        except Exception as e:
            self.loss_out.insert(tk.END, f"\n[ERROR] {e}\n")

    def _plot_loss_curves(self, f, T, S, D, bottom, wind, sea):
        self.fig_loss.clear()
        ax1 = self.fig_loss.add_subplot(211)
        ax2 = self.fig_loss.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor("#040410")
            ax.tick_params(colors="white")
            for s in ax.spines.values():
                s.set_color("#333355")
            ax.grid(True, alpha=0.15, color="#334455")

        r_arr = np.linspace(1, max(D * 30, 500), 300)

        TL_sph  = [20*math.log10(r) for r in r_arr]
        TL_cyl  = [10*math.log10(r) for r in r_arr]
        alpha   = francois_garrison_absorption(f, T, S, D)
        TL_abs  = [alpha*r for r in r_arr]
        TL_tot  = [total_transmission_loss(r, f, T, S, D, 5, D/2,
                    bottom, wind, sea)['total'] for r in r_arr]

        ax1.plot(r_arr, TL_sph, color="#33ccff", lw=2, label="Spherical (20 log R)")
        ax1.plot(r_arr, TL_cyl, color="#5577ff", lw=2, ls="--", label="Cylindrical (10 log R)")
        ax1.plot(r_arr, TL_abs, color="#cc44ff", lw=2, label="Absorption only")
        ax1.plot(r_arr, TL_tot, color="#ff4444", lw=2.5, label="Total TL")
        ax1.set_xlabel("Range (m)", color="white")
        ax1.set_ylabel("TL (dB)", color="white")
        ax1.set_title(f"TL Components @ {f/1000:.1f} kHz — {bottom} bottom",
                     color="white", fontsize=9)
        ax1.legend(facecolor="#0d0d25", labelcolor="white", fontsize=8)

        # Bottom loss vs grazing angle for all sediment types
        angles = np.linspace(0.5, 89, 200)
        for bt, col in [('sand','#ffcc33'),('mud','#44ff88'),
                        ('clay','#33aaff'),('rock','#ff5533'),('silt','#cc99ff')]:
            bl = [bottom_reflection_loss(f, a, bt) for a in angles]
            ax2.plot(angles, bl, color=col, lw=1.8, label=bt)
        ax2.set_xlabel("Grazing Angle (°)", color="white")
        ax2.set_ylabel("Bottom Loss (dB)", color="white")
        ax2.set_title("Bottom Reflection Loss vs Grazing Angle", color="white", fontsize=9)
        ax2.legend(facecolor="#0d0d25", labelcolor="white", fontsize=8,
                  ncol=3, loc="upper right")

        self.fig_loss.tight_layout()
        self.fig_loss.patch.set_facecolor("#040410")
        self.canvas_loss.draw()

    # ────────────────────────────────────────────────────────────
    #  ENV FILE / DIRECTORY ACTIONS
    # ────────────────────────────────────────────────────────────

    def _preview_env(self):
        freq = int(self.freq_var.get() or 25000)
        p = (get_shallow_water_params(freq) if self.is_shallow
             else get_deep_water_params(freq))
        path = os.path.join(self.work_dir, 'niot_run.env')
        generate_env_file(path, p)
        self.file_view.delete("1.0", tk.END)
        self.file_view.insert(tk.END, f"=== {path} ===\n\n" + read_text_file(path))
        self.nb.select(self.tab_files)
        self.log("[ENV] .env file previewed")

    def _show_file(self, suffix):
        path = os.path.join(self.work_dir, f"niot{suffix}")
        self.file_view.delete("1.0", tk.END)

        if not os.path.exists(path):
            self.file_view.insert(tk.END,
                f"File not found: {path}\nRun Bellhop first.\n\n"
                "Files currently in work dir:\n")
            for fn in sorted(os.listdir(self.work_dir)):
                self.file_view.insert(tk.END, f"  {fn}\n")
            return

        if suffix.endswith('.shd'):
            rrs, rds, TL, title = read_shd_file(path)
            if TL is not None:
                info = (f"=== {path} (binary SHD) ===\n"
                        f"Title   : {title}\n"
                        f"Ranges  : {len(rrs)} pts  [{rrs[0]:.4f}–{rrs[-1]:.4f}] km\n"
                        f"Depths  : {len(rds)} pts  [{rds[0]:.2f}–{rds[-1]:.2f}] m\n"
                        f"TL range: {np.nanmin(TL):.2f}–{np.nanmax(TL):.2f} dB\n"
                        f"Grid    : {TL.shape[0]}×{TL.shape[1]}\n\n"
                        "→ See 'TRANS. LOSS' tab for the TL plot.")
                self.file_view.insert(tk.END, info)
            else:
                self.file_view.insert(tk.END, f"SHD read error: {title}")
        else:
            content = read_text_file(path)
            sz = os.path.getsize(path)
            self.file_view.insert(tk.END,
                f"=== {path}  ({sz} bytes) ===\n\n{content}")

    def _list_work_dir(self):
        self.file_view.delete("1.0", tk.END)
        self.file_view.insert(tk.END, f"Work Directory: {self.work_dir}\n{'─'*60}\n")
        for fn in sorted(os.listdir(self.work_dir)):
            fp   = os.path.join(self.work_dir, fn)
            size = os.path.getsize(fp)
            self.file_view.insert(tk.END, f"  {fn:<36} {size:>9,} bytes\n")

    def _open_work_dir(self):
        self._list_work_dir()
        self.nb.select(self.tab_files)

    # ────────────────────────────────────────────────────────────
    #  LIVE ACOUSTIC CHANNEL UPDATE
    # ────────────────────────────────────────────────────────────

    def _update_acoustic_live(self):
        try:
            p   = self.params
            f   = p['freq']
            r   = (p['r_max_km'] * 1000 / 2)
            loss = total_transmission_loss(r, f, p['T'], p['S'], p['depth'],
                                           p['source_depths'][0], p['depth']/2,
                                           p['bottom_type'], p['wind_ms'], p['sea_state'])
            N   = ambient_noise_wenz(f, p['sea_state'])
            SNR = 180.0 - loss['total'] - N

            self.acv_tl.set(f"TL: {loss['total']:.2f} dB")
            self.acv_snr.set(f"SNR: {SNR:.1f} dB @ {r:.0f} m")
            self.acv_range.set(f"Range: 0 – {p['r_max_km']*1000:.0f} m")
            self.acv_alpha.set(f"α: {loss['absorption_dBkm']:.4f} dB/km")

            self.log(f"[ACOUSTIC] TL={loss['total']:.2f} dB | SNR={SNR:.1f} dB | "
                    f"α={loss['absorption_dBkm']:.4f} dB/km | "
                    f"Bounces={loss['bottom_bounces']} | "
                    f"Grazing={loss['grazing_angle']:.1f}°")
        except Exception as e:
            self.log(f"[ERROR] Acoustic update: {e}")

    # ────────────────────────────────────────────────────────────
    #  USBL TX / RX  (from original usblgui_v1.py)
    # ────────────────────────────────────────────────────────────

    def _start_tx(self):
        if self.tx_running:
            return
        try:
            self._mav_tx = mavutil.mavlink_connection(self.MAVLINK_UDP)
            self.tx_running = True
            self.btn_tx.config(bg="#446644", text="TX ● Running")
            threading.Thread(target=self._tx_loop, daemon=True).start()
            self.log("[TX] Started — waiting for MAVLink messages")
        except Exception as e:
            self.log(f"[TX ERROR] Cannot connect MAVLink: {e}")

    def _tx_loop(self):
        while self.tx_running:
            try:
                msg = self._mav_tx.recv_match(blocking=True, timeout=1.0)
                if not msg or msg.get_type() not in self.ALLOWED_MSGS:
                    continue

                t = msg.get_type()
                if t == "HEARTBEAT":
                    self.txv_mode.set(f"Mode: {self._mav_tx.flightmode}")
                if t == "GLOBAL_POSITION_INT":
                    lat = msg.lat / 1e7; lon = msg.lon / 1e7
                    self.txv_pos.set(f"Lat:{lat:.5f}  Lon:{lon:.5f}")

                payload = zlib.compress(msg.get_msgbuf(),
                                        self.COMPRESS_LVL)[:self.MAX_PAYLOAD]
                pkt = f"B|1|0|{self.msg_counter}|".encode() + payload

                if self._send_usbl(pkt):
                    self._send_modems(pkt)
                    self.msg_counter += 1
                    self.txv_count.set(f"Packets TX: {self.msg_counter}")
                    self.txv_usbl.set("USBL: ONLINE ✓")
                    self.log(f"[TX] Broadcast #{self.msg_counter}  {t}")
                else:
                    self.txv_usbl.set("USBL: OFFLINE ✗")

                time.sleep(0.05)
            except Exception as e:
                if self.tx_running:
                    self.log(f"[TX ERR] {e}")

    def _send_usbl(self, pkt):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((self.USBL_IP, self.USBL_PORT))
            s.send(pkt); s.close()
            return True
        except:
            return False

    def _send_modems(self, pkt):
        for port in self.MODEM_PORTS:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect((self.MODEM_IP, port))
                cmd = b"AT*SENDIM," + str(len(pkt)).encode() + b"," + pkt + b"\n"
                s.send(cmd); s.close()
                if port == 9001: self.modv1.set("Port 9001: ● ACTIVE")
                if port == 9002: self.modv2.set("Port 9002: ● ACTIVE")
            except:
                if port == 9001: self.modv1.set("Port 9001: ○ OFF")
                if port == 9002: self.modv2.set("Port 9002: ○ OFF")

    def _start_rx(self):
        if self.rx_running:
            return
        self.rx_running = True
        self.btn_rx.config(bg="#334488", text="RX ● Running")
        for port in self.MODEM_PORTS:
            threading.Thread(target=self._rx_server, args=(port,), daemon=True).start()
        self.log("[RX] Modem servers started")

    def _rx_server(self, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(5)
        self.log(f"[RX] Listening on port {port}")

        while self.rx_running:
            try:
                client, _ = srv.accept()
                data = client.recv(4096)
                if data:
                    decoded = self._decode_packet(data)
                    self.rx_counts[port] += 1
                    total = sum(self.rx_counts.values())
                    self.rxv_last.set(f"Last: {decoded}")
                    self.rxv_count.set(f"Received: {total}")
                    self.log(f"[RX:{port}] #{self.rx_counts[port]} → {decoded}")
                client.close()
            except:
                pass

    def _decode_packet(self, raw):
        try:
            zi = raw.find(b'\x78\x9c')
            if zi == -1:
                return "raw/unknown"
            dec  = zlib.decompress(raw[zi:])
            mav  = mavutil.mavlink_connection('udpin:localhost:0')
            msgs = mav.mav.parse_buffer(dec)
            return msgs[0].get_type() if msgs else "unknown MAVLink"
        except Exception as e:
            return f"decode-err: {e}"


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = USBLBellhopApp(root)
    root.mainloop()
