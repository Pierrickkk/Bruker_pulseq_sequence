"""Simplified GRE sequence with MRE motion encoding gradients (single frequency)."""

import time
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_trapezoid import make_trapezoid

# ======
# FLAGS
# ======
FLAG_SHOW_PLOTS   = True
FLAG_TEST_REPORT  = True
FLAG_TIMING_CHECK = True
FLAG_WRITE_SEQ    = False

# ======
# SYSTEM
# ======
system = pp.Opts(
    max_grad=34, grad_unit='mT/m',
    max_slew=100, slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
seq = pp.Sequence(system=system)

# ======
# GEOMETRY
# ======
fov             = 220e-3   # field of view [m]
n_x             = 64
n_y             = 64
slice_thickness = 5e-3     # [m]

# ======
# RF PULSE
# ======
rf_angle        = 10       # flip angle [deg]
rf_duration     = 1.28e-3  # [s]
rf_spoiling_inc = 117      # RF spoiling phase increment [deg], 0 = disabled

# ======
# MRE PARAMETERS
# ======
mre_exc_freq       = 1000.0        # single mechanical excitation frequency [Hz]
mre_wave_period    = 1 / mre_exc_freq
mre_n_timesteps    = 4             # number of phase offsets (time steps) over one wave period
mre_meg_cycles     = 2             # number of MEG cycles (bipolar gradient pairs)
mre_meg_orientations =  ['x']        #['x', 'y', 'z']
mre_meg_duration   = mre_wave_period * 2        # total MEG duration [s] (adjusted automatically if needed) (Must be = N*mre_wave_period)
mre_exp_number     = 10            # experiment number encoded in trigger pulse width
#TODO tester si meg duration est bien un multiple de mre_wave_period

# ======
# EVENTS
# ======
# Slice-selective sinc pulse
rf, gz, gzr = pp.make_sinc_pulse(
    flip_angle=rf_angle * np.pi / 180,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    use='excitation',
)
gzr.duration = 0.00084
# Readout gradient and ADC
delta_k  = 1 / fov
gx       = pp.make_trapezoid(channel='x', flat_area=n_x * delta_k,
                              flat_time=system.grad_raster_time * n_x, system=system)
adc      = pp.make_adc(num_samples=n_x, duration=gx.flat_time,
                       delay=gx.rise_time, system=system)
gx_pre   = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=0.00042,system=system)

# Phase encoding: one trapezoid per PE step, calculated on the fly
phase_areas = (np.arange(n_y) - n_y // 2) * delta_k
gy_pre_ref  = pp.make_trapezoid(channel='y', area=np.max(np.abs(phase_areas)), system=system)
gy_dur      = pp.calc_duration(gy_pre_ref)*2

# Motion encoding gradients (zeroth-order moment nulling: +/-)
# Each MEG cycle = one positive lobe + one negative lobe of duration mre_meg_duration/2
meg_lobe_dur = mre_meg_duration / (2 * mre_meg_cycles)
meg_lobe = make_trapezoid(channel='z', amplitude=system.max_grad / 2,
                          duration=meg_lobe_dur, system=system)

# Spoilers
gx_spoil = make_trapezoid(channel='x', area=2 * n_x * delta_k, system=system)
gz_spoil = make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# ======
# TIMING
# ======
#time_label = 1e-5  # short block for label insertion [s]

total_meg_dur = pp.calc_duration(meg_lobe) * 2 * mre_meg_cycles

min_TE = (
    pp.calc_duration(gz) / 2       # half RF pulse
    + total_meg_dur                # all MEG lobes
    + pp.calc_duration(gzr, gx_pre, gy_pre_ref)
    + pp.calc_duration(gx) / 2    # half readout
)
min_TE = np.ceil(min_TE / system.grad_raster_time) * system.grad_raster_time
min_TE = np.ceil(min_TE * 1e8) / 1e8

min_TR = (
    pp.calc_duration(gz)
    + total_meg_dur
    + pp.calc_duration(gzr, gx_pre, gy_pre_ref)
    + pp.calc_duration(gx, adc)
    + pp.calc_duration(gx_spoil, gy_pre_ref, gz_spoil)
    #+ time_label
)
min_TR = np.ceil(min_TR / system.grad_raster_time) * system.grad_raster_time
min_TR = np.ceil(min_TR / mre_wave_period) * mre_wave_period  # ensure TR is a multiple of the wave period
min_TR = np.ceil(min_TR / 1e8) * 1e8

# No TE/TR delay (minimal TE and TR)
delay_TE = 0.0
delay_TR = 0.0
final_TE = delay_TE + min_TE
final_TR = delay_TR + min_TR

print(f'min TE = {min_TE * 1e3:.2f} ms  |  min TR = {min_TR * 1e3:.2f} ms')

# ======
# OUTPUT PATH & FILENAME
# ======
output_path = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/output"
filename = (f'{time.strftime("%Y%m%d")}_mre_notrig3_gre'
            f'_{n_x}nx_{n_y}ny_{int(mre_exc_freq)}Hz'
            f'_{mre_n_timesteps}ts_{mre_meg_cycles}cyc')

# ======
# SEQUENCE CONSTRUCTION
# ======
# Experiment trigger (encodes experiment number in pulse width)
exp_trig_dur = 20e-6 + mre_exp_number * 10e-6
#seq.add_block(make_digital_output_pulse('ext1', duration=exp_trig_dur))
#seq.add_block(pp.make_delay(50e-6))

# --- ID triggers (encode wave period index) ---
#seq.add_block(make_digital_output_pulse('ext1', duration=1100e-6))
#seq.add_block(pp.make_delay(100e-6))
#seq.add_block(make_digital_output_pulse('ext1', duration=1200e-6))
#seq.add_block(pp.make_delay(100e-6))
#seq.add_block(make_digital_output_pulse('ext1', duration=1300e-6))
#seq.add_block(pp.make_delay(1200e-6))

# Vibration start trigger
#seq.add_block(make_digital_output_pulse('ext1', duration=10e-6))
#seq.add_block(pp.make_delay(990e-6))

vibration_start_time = sum(seq.block_durations.values())

# Calibration trigger
#seq.add_block(make_digital_output_pulse('ext1', duration=1))
#seq.add_block(pp.make_delay(1))

# RF spoiling state
rf_phase = 0.0
rf_inc   = 0.0

# --- Loop over MEG orientations ---
for n_dim, meg_orientation in enumerate(mre_meg_orientations):
    #seq.add_block(pp.make_delay(time_label),
                 # make_label(type='SET', label='SET', value=n_dim))

    meg_lobe.channel = meg_orientation

    # --- Loop over MRE time steps (phase offsets) ---
    for idx_timestep in range(mre_n_timesteps):
        #seq.add_block(pp.make_delay(time_label),
                      #make_label(type='SET', label='PHS', value=idx_timestep))

        # Target phase offset within one wave period
        wanted_phase = idx_timestep / mre_n_timesteps * mre_wave_period

        # --- Loop over phase encoding steps ---
        for idx_y in range(n_y):
            #seq.add_block(pp.make_delay(time_label),
            #              make_label(type='SET', label='LIN', value=idx_y))

            # Current phase position in the wave cycle
            current_phase = np.mod(
                sum(seq.block_durations.values()) - vibration_start_time,
                mre_wave_period,
            )

            # Delay to reach the desired phase offset
            if wanted_phase >= current_phase:
                delay_sync = wanted_phase - current_phase
            else:
                delay_sync = mre_wave_period - (current_phase - wanted_phase)
            delay_sync = np.ceil(delay_sync / system.grad_raster_time) * system.grad_raster_time
            seq.add_block(make_delay(delay_sync))

            # RF spoiling
            rf.phase_offset  = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc   = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)

            # Excitation
            seq.add_block(rf, gz)

            # MEG: mre_meg_cycles bipolar pairs (+/-)
            for _ in range(mre_meg_cycles):
                meg_lobe.amplitude = abs(meg_lobe.amplitude)
                seq.add_block(meg_lobe)
                meg_lobe.amplitude = -abs(meg_lobe.amplitude)
                seq.add_block(meg_lobe)

            # TE delay (0 for minimal TE)
            if delay_TE > 0:
                seq.add_block(make_delay(delay_TE))

            # Phase encoding gradient for current PE step
            gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[idx_y],
                                        duration=gy_dur, system=system)

            # Pre-winders + slice rewinder
            seq.add_block(gx_pre, gy_pre, gzr)

            # Readout
            seq.add_block(gx, adc)

            # Rewind PE + spoil
            gy_pre.amplitude *= -1
            seq.add_block(gx_spoil, gy_pre, gz_spoil)

            # TR delay (0 for minimal TR)
            if delay_TR > 0:
                seq.add_block(make_delay(delay_TR))

            # Keep-alive trigger
            #seq.add_block(make_digital_output_pulse('ext1', duration=10e-6))

# End-of-experiment trigger
#seq.add_block(pp.make_delay(100e-3))
#seq.add_block(make_digital_output_pulse('ext1', duration=1e-3))
#seq.add_block(pp.make_delay(1e-3))

# ======
# CHECKS
# ======
if FLAG_TIMING_CHECK and not FLAG_TEST_REPORT:
    ok, error_report = seq.check_timing()
    print('\nTiming check passed.' if ok else f'\nTiming check failed:\n{error_report}')

if FLAG_TEST_REPORT:
    print('\nAdvanced test report:')
    print(seq.test_report())

# ======
# EXPORT
# ======
seq.set_definition('FOV',                [fov, fov, slice_thickness])
seq.set_definition('Name',               'gre')
seq.set_definition('Flipangle',          rf_angle)
seq.set_definition('TE',                 final_TE)
seq.set_definition('TR',                 final_TR)
seq.set_definition('k_space_encoding1',  n_x)
seq.set_definition('number_of_readouts', n_y)
seq.set_definition('sampling_scheme',    'cartesian')
seq.set_definition('slices',             1)

if FLAG_WRITE_SEQ:
    out_file = f'{output_path}/{filename}'
    print(f"\nSaving '{filename}.seq'")
    seq.write(out_file, create_signature=True)

# ======
# PLOTS
# ======

if FLAG_SHOW_PLOTS:
    seq.plot(time_range=[0, 0.1])
    k_traj_adc, k_traj, *_ = seq.calculate_kspace()
    plt.figure()
    N3=64*16*3
    plt.plot(k_traj[0,1:N3], k_traj[1,1:N3], 'b')
    plt.plot(k_traj_adc[0,1:N3], k_traj_adc[1,1:N3], '.r', markersize=3)
    plt.title('k-space trajectory')
    plt.show()