"""GRE sequence with MRE motion encoding gradients."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_label import make_label
from pypulseq.make_trapezoid import make_trapezoid

# choose settings
FLAG_SHOW_PLOTS = True  # toggle plotting of gradients/trajectory etc
FLAG_TEST_REPORT = True  # toggle advanced test report including timing check (SLOW)
FLAG_TIMING_CHECK = True  # toggle timing check (SlOW)
FLAG_WRITE_SEQ = True  # toggle writing of seq-file
# Define system limits
system = pp.Opts(
    max_grad=34,
    grad_unit='mT/m',
    max_slew=100,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    #block_duration_raster=10e-6,
    #grad_raster_time=10e-5,
)

# Create PyPulseq sequence object
seq = pp.Sequence(system=system)

# define geometry parameters
fov = 220e-3
n_x = 64
n_y = 64
res = fov / n_x  # spatial resolution [m]
slice_thickness = 5e-3  # slice thickness [m]

# define rf pulse parameters
rf_angle = 10  # flip angle of excitation pulse [°]
rf_duration = 1.28e-3  # duration of excitation pulse [s]
rf_spoiling_inc = 117  # rf spoiling phase increment. Choose 0 to disable rf spoiling. [°]

# define timing parameters
TR: float | None = None  # repetition time TR [s]. Set to None for minimal TR
TE: float | list | tuple | None = None  # echo time TE [s]. Set to None for minimal TE
n_TE = len(TE) if isinstance(TE, list | tuple) else 1

# Elastographie parameters
mre_exp_number = 10
mre_exc_freq = [1000]  # [30.03, 40.0, 50.0, 60.24, 70.42, 80.0, 90.09, 100.0]
mre_wave_period = [1 / freq for freq in mre_exc_freq]
mre_n_timesteps = 8
mre_meg_orientations = ['x', 'y', 'z']
mre_meg_duration = 20e-3
mre_meg_moment_nulling = 0

# Create alpha-degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
    flip_angle=rf_angle * np.pi / 180,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    use='excitation',
)

# Define readout gradient(s) and ADC
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_k, flat_time=system.grad_raster_time * n_x, system=system)
adc = pp.make_adc(num_samples=n_x, dwell=system.grad_raster_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, system=system)
gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, system=system)

# Calculate phase encoding gradient areas and create largest phase encoding gradient
phase_areas = (np.arange(n_y) - n_y // 2) * delta_k
gy_pre = pp.make_trapezoid(channel='y', area=np.max(np.abs(phase_areas)), system=system)
gy_dur = pp.calc_duration(gy_pre)

# Create motion encoding gradients
# TODO: Automatic duration adaptation to vibration frequency
# TODO: MEG Repetitions
meg_grad_half_dur = make_trapezoid(
    channel='z', amplitude=system.max_grad / 2, duration=mre_meg_duration / 2, system=system
)
meg_grad_quarter_dur = make_trapezoid(
    channel='z', amplitude=system.max_grad / 2, duration=mre_meg_duration / 4, system=system
)

# Create gradient spoilers
gx_spoil = make_trapezoid(channel='x', area=2 * n_x * delta_k, system=system)
gz_spoil = make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing parameters
min_TE = (
    pp.calc_duration(gz) / 2  # half duration of rf pulse
    + pp.calc_duration(meg_grad_half_dur) * 2
    + pp.calc_duration(gzr, gx_pre, gy_pre)
    + pp.calc_duration(gx) / 2  # half readout gradient
)
min_TE = np.ceil(min_TE / system.grad_raster_time) * system.grad_raster_time  # put on raster
min_TE = np.ceil(min_TE * 1e8) / 1e8  # round to 2 decimal values in ms

# calculate minimum TR time
time_label = 1e-5  # time for label setting
min_TR = (
    pp.calc_duration(gz)  # rf pulse
    + pp.calc_duration(meg_grad_half_dur) * 2
    + pp.calc_duration(gzr, gx_pre, gy_pre)
    + pp.calc_duration(gx, adc)  # readout gradient
    + pp.calc_duration(gx_spoil, gy_pre, gz_spoil)  # gradient spoiling
    + time_label  # label setting
)
min_TR = np.ceil(min_TR / system.grad_raster_time) * system.grad_raster_time  # put on raster
min_TR = np.ceil(min_TR * 1e8) / 1e8  # round to 2 decimal values in ms

# calculate echo time delay (delay_TE)
if TE is None:
    delay_TE = 0
elif isinstance(TE, float):
    delay_TE = np.ceil((TE - min_TE) / system.grad_raster_time) * system.grad_raster_time
    if not delay_TE >= 0:
        raise ValueError(f'TE must be larger than {min_TE * 1000:.2f} ms. Current value is {TE * 1000:.2f} ms.')
elif isinstance(TE, list | tuple):
    delay_TE = [np.ceil((te - min_TE) / system.grad_raster_time) * system.grad_raster_time for te in TE]
    if not all(delay >= 0 for delay in delay_TE):
        raise ValueError(
            f'All TE must be larger than {min_TE * 1000:.2f} ms. Current values are {[te * 1000 for te in TE]} ms.'
        )

# calculate final TE for consistency check and seq-file header
final_TE = delay_TE + min_TE

# calculate minimum TR for given TE value(s)
min_TR_given_TE = np.ceil((min_TR + delay_TE) / system.grad_raster_time) * system.grad_raster_time

# calculate repetition time delay depending TE/TR settings
# TR None and single TE value -> no delay
if TR is None and isinstance(min_TR_given_TE, float):
    delay_TR: float = 0  # type: ignore
# TR None and multiple TE values -> multiple TR delays
elif TR is None and isinstance(min_TR_given_TE, np.ndarray):
    delay_TR: np.ndarray = -min_TR_given_TE + min_TR_given_TE.max()  # type: ignore
    assert all(delay >= 0 for delay in delay_TR)
# TR and TE single value -> single TR delay
elif isinstance(TR, float) and isinstance(min_TR_given_TE, float):
    delay_TR: float = np.ceil((TR - min_TR_given_TE) / system.grad_raster_time) * system.grad_raster_time  # type: ignore
    if not delay_TR >= 0:
        raise ValueError(
            f'TR must be larger than {min_TR_given_TE * 1000:.2f} ms. Current value is {TR * 1000:.2f} ms.'
        )
# TR single value and multiple TE values -> multiple TR delays
elif isinstance(min_TR_given_TE, np.ndarray):
    delay_TR: np.ndarray = np.ceil((TR - min_TR_given_TE) / system.grad_raster_time) * system.grad_raster_time  # type: ignore
    if not all(delay >= 0 for delay in delay_TR):
        raise ValueError(
            f'TR must be larger than {min_TR_given_TE.max() * 1000:.2f} ms. Current value is {TR * 1000:.2f} ms.'
        )

# calculate final TR for consistency check and seq-file header
final_TR = min_TR_given_TE + delay_TR
final_TR = np.unique(final_TR)
assert final_TR.size == 1
final_TR = final_TR.item()

# define full filename
filename = f'{time.strftime("%Y%m%d")}_mre_gre'
filename += f'_{n_x}nx_{n_y}ny_{len(mre_exc_freq)}freqs'

# create folder for seq and header file
output_path = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/output"

# delete existing header file
#if (output_path / f'{filename}_header.h5').exists():
 #   (output_path / f'{filename}_header.h5').unlink()

# select experiment trigger
exp_trig_dur = 20e-6 + mre_exp_number * 10e-6
trig = make_digital_output_pulse('ext1', duration=exp_trig_dur)
seq.add_block(trig)
seq.add_block(pp.make_delay(50e-6))

# loop over MRE frequencies / wave periods
for i_wp in range(len(mre_wave_period)):
    # set REP label (MRE frequency / wave period)
    seq.add_block(make_label(type='SET', label='REP', value=i_wp))

    # id trigger 1
    seq.add_block(make_digital_output_pulse('ext1', duration=1100e-6))
    seq.add_block(pp.make_delay(100e-6))

    # id trigger 2
    seq.add_block(make_digital_output_pulse('ext1', duration=1200e-6))
    seq.add_block(pp.make_delay(100e-6))

    # id trigger 3
    seq.add_block(make_digital_output_pulse('ext1', duration=1300e-6))

    # delay after id trigger
    seq.add_block(pp.make_delay(1200e-6))

    # short trigger / start of vibration ?!
    seq.add_block(make_digital_output_pulse('ext1', duration=10e-6))
    seq.add_block(pp.make_delay(990e-6))

    # save start time
    vibration_start_time = sum(seq.block_durations.values())

    # calibration trigger and delay
    seq.add_block(make_digital_output_pulse('ext1', duration=1))
    seq.add_block(pp.make_delay(1))

    # set initial rf phase offset
    rf_phase = 0
    rf_inc = 0

    # loop over MEG orientations
    for n_dim, meg_orientation in enumerate(mre_meg_orientations):
        # set SET label (MRE orientation)
        seq.add_block(pp.make_delay(time_label), make_label(type='SET', label='SET', value=n_dim))

        # set MEG orientation
        meg_grad_half_dur.channel = meg_orientation
        meg_grad_quarter_dur.channel = meg_orientation

        # loop over MRE time steps
        for idx_timestep in range(mre_n_timesteps):
            # set PHS label (time step)
            seq.add_block(pp.make_delay(time_label), make_label(type='SET', label='PHS', value=idx_timestep))

            # calculate wanted wave period for current time step
            wanted_wave_period = idx_timestep / mre_n_timesteps * mre_wave_period[i_wp]

            # loop over phase encoding steps
            for idx_y in range(n_y):
                # set LIN label (interleave)
                seq.add_block(pp.make_delay(time_label), make_label(type='SET', label='LIN', value=idx_y))

                # get current wave period
                current_wave_period = np.mod(
                    sum(seq.block_durations.values()) - vibration_start_time, mre_wave_period[i_wp]
                )

                # calculate delay to synchronize
                if wanted_wave_period == current_wave_period:
                    delay_to_synchronize = 0
                elif wanted_wave_period > current_wave_period:
                    delay_to_synchronize = wanted_wave_period - current_wave_period
                elif wanted_wave_period < current_wave_period:
                    delay_to_synchronize = mre_wave_period[i_wp] - (current_wave_period - wanted_wave_period)

                # round and add delay to synchronize
                delay_to_synchronize = round(delay_to_synchronize / system.grad_raster_time) * system.grad_raster_time
                seq.add_block(make_delay(delay_to_synchronize))

                # loop over echo times
                for c in range(n_TE):
                    # todo: set echo time label
                    # set current phase offset of rf and adc
                    rf.phase_offset = rf_phase / 180 * np.pi
                    adc.phase_offset = rf_phase / 180 * np.pi

                    # calculate phase offset for next shot
                    rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
                    rf_phase = np.mod(rf_phase + rf_inc, 360.0)

                    # add slice selective excitation pulse
                    seq.add_block(rf, gz)

                    # add motion encoding gradients
                    if mre_meg_moment_nulling == 0:  # zeroth order moment nulling
                        # positive lobe
                        meg_grad_half_dur.amplitude = abs(meg_grad_half_dur.amplitude)
                        seq.add_block(meg_grad_half_dur)

                        # negative lobe
                        meg_grad_half_dur.amplitude = -1 * abs(meg_grad_half_dur.amplitude)
                        seq.add_block(meg_grad_half_dur)
                    elif mre_meg_moment_nulling == 1:  # first order moment nulling
                        # positive lobe
                        meg_grad_quarter_dur.amplitude = abs(meg_grad_quarter_dur.amplitude)
                        seq.add_block(meg_grad_quarter_dur)

                        # negative lobe
                        meg_grad_half_dur.amplitude = -1 * abs(meg_grad_half_dur.amplitude)
                        seq.add_block(meg_grad_half_dur)

                        # positive lobe
                        meg_grad_quarter_dur.amplitude = abs(meg_grad_quarter_dur.amplitude)
                        seq.add_block(meg_grad_quarter_dur)

                    # add echo time delay
                    current_delay_TE = delay_TE[c] if isinstance(delay_TE, list | tuple) else delay_TE
                    if current_delay_TE > 0:
                        seq.add_block(make_delay(current_delay_TE))

                    # adjust phase encoding gradient to current phase encoding step
                    gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[idx_y], duration=gy_dur, system=system)

                    # add readout pre-winder, phase encoding and slice rewinder gradients
                    seq.add_block(gx_pre, gy_pre, gzr)

                    # add readout gradient and ADC
                    seq.add_block(gx, adc)

                    # add rewinder and spoiler gradients
                    gy_pre.amplitude *= -1
                    seq.add_block(gx_spoil, gy_pre, gz_spoil)

                    # add repetition time delay
                    current_delay_TR = delay_TR[c] if isinstance(delay_TR, list | tuple) else delay_TR
                    if current_delay_TR > 0:
                        seq.add_block(make_delay(current_delay_TR))

                    # add keep alive trigger
                    seq.add_block(make_digital_output_pulse('ext1', duration=10e-6))

    # add delay before next sub-experiment trigger to avoid overlap with keep alive trigger
    seq.add_block(pp.make_delay(100e-3))

    # add sub-experiment trigger and delay
    seq.add_block(make_digital_output_pulse('ext1', duration=1e-3))
    seq.add_block(pp.make_delay(1e-3))

# check timing of the sequence
if FLAG_TIMING_CHECK and not FLAG_TEST_REPORT:
    ok, error_report = seq.check_timing()
    if ok:
        print('\nTiming check passed successfully')
    else:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)

# show advanced rest report
if FLAG_TEST_REPORT:
    print('\nCreating advanced test report...')
    print(seq.test_report())

# prepare sequence export
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.set_definition(key='Flipangle', value=rf_angle)
seq.set_definition(key='TE', value=final_TE)
seq.set_definition(key='TR', value=final_TR)
seq.set_definition(key='k_space_encoding1', value=n_x)
seq.set_definition(key='name', value='grpe')
seq.set_definition(key='number_of_readouts', value=n_y)
seq.set_definition(key='sampling_scheme', value='cartesian')
seq.set_definition(key='slices', value=1)

# save seq-file
print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
if FLAG_WRITE_SEQ:
    seq.write(str(output_path + '/' + filename), create_signature=True)

if FLAG_SHOW_PLOTS:
    seq.plot()

    # calculate k-space trajectory from sequence
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # plot k-space trajectory
    plt.figure()
    plt.plot(k_traj[0], k_traj[1], 'b')
    plt.plot(k_traj_adc[0], k_traj_adc[1], '.', color='red', markersize=3)
    plt.show()