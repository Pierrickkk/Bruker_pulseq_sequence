"""Variable density spiral sequence."""

import time
from pathlib import Path


import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from utils.create_ismrmrd_header import create_hdr
from utils.vds import vds
from utils.write_seq_definitions import write_seq_definitions

# choose flags
FLAG_TRIGGER = True  # toggle triggered start of the sequence. The delay between trigger and start is defined below.
FLAG_GOLDEN_ANGLE = True  # toggle use of golden angle
FLAG_PLOTS = True  # toggle plotting of gradients/trajectory etc
FLAG_TESTREPORT = True  # toggle advanced test report including timing check (SLOW)
FLAG_TIMINGCHECK = True  # toggle timing check (SlOW)
FLAG_WRITE_SEQ = False # Toggle writing seq file
# define filename
filename = f'{time.strftime("%Y%m%d")}_spiral_2D'

# define geometry parameters
fov = 256e-3  # field of view [m]
n_x = 256  # number of points per spoke
slice_thickness = 8e-3  # slice thickness [m]
res = fov / n_x  # spatial resolution [m]

# define number of spirals and variable density parameter
n_spirals = 32  # number of interleaves
fov_scale = 6
n_spirals_for_vds_calc = 12  # 0: 86, 1: 72, 2: 60, 3: 48, 4: 39, 5: 31, 6: 24, 7: 19
fov_coeff = [fov, -fov_scale / 8 * fov]  # FOV decreases linearly from fov_coeff[0] to fov_coeff[0]-fov_coeff[1].

# set repetition time
TR = None  # repetition time [s]. Set to None for minimum TR

# define rf pulse parameters
rf_angle = 15  # flip angle of excitation pulse [°]
rf_duration = 1.28e-3  # duration of excitation pulse [s]
rf_bwt_product = 4  # bandwidth time product of rf pulses. MUST BE THE SAME FOR ALL PULSES !!!
rf_spoiling_inc = 0  # rf spoiling phase increment. Choose 0 to disable rf spoiling. [°]

# create Pypulseq Sequence object and set system limits
system = pp.Opts(
    max_grad=30,
    grad_unit='mT/m',
    max_slew=100,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

seq = pp.Sequence(system=system)

# calculate angular increment
delta_angle = 2 * np.pi * (1 - 2 / (1 + np.sqrt(5))) if FLAG_GOLDEN_ANGLE else 2 * np.pi / n_spirals

# define set label time
time_label = 1e-5

# create slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
    flip_angle=rf_angle * np.pi / 180,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=rf_bwt_product,
    system=system,
    return_gz=True,
)

# calculate spiral trajectory
r_max = 0.5 / fov * n_x  # [1/m]
k, g, s, timing, r, theta = vds(
    smax=system.max_slew * 0.9,
    gmax=system.max_grad * 0.9,
    T=system.grad_raster_time,
    N=n_spirals_for_vds_calc,
    Fcoeff=fov_coeff,
    rmax=r_max,
    oversampling=12,
)

# Pre-calculate the spiral gradient waveforms, k-space trajectories, and rewinders
n_points_g = np.shape(g)[0]
n_points_k = np.shape(k)[0]

# calculate ADC
adc_total_samples = np.shape(g)[0] - 1
# Ensure that adc_total_samples does not exceed the Siemenes ADC limit. Check pulseq documentation for details.
assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
adc_dwell = system.grad_raster_time
adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, system=system)

spiral_readout_grad = np.zeros((n_spirals, 2, n_points_g))
spiral_trajectory = np.zeros((n_spirals, 2, n_points_k))
gx_readout_list = []
gy_readout_list = []
gx_rewinder_list = []
gy_rewinder_list = []
rewinder_duration = 0

for n in range(n_spirals):
    delta = delta_angle * n
    exp_delta = np.exp(1j * delta)
    exp_delta_pi = np.exp(1j * (delta + np.pi))

    spiral_readout_grad[n, 0, :] = np.real(g * exp_delta)
    spiral_readout_grad[n, 1, :] = np.imag(g * exp_delta)
    spiral_trajectory[n, 0, :] = np.real(k * exp_delta_pi)
    spiral_trajectory[n, 1, :] = np.imag(k * exp_delta_pi)

    gx_readout = pp.make_arbitrary_grad(
        channel='x', waveform=spiral_readout_grad[n, 0], first=0, system=system, delay=adc.delay
    )
    gy_readout = pp.make_arbitrary_grad(
        channel='y', waveform=spiral_readout_grad[n, 1], first=0, system=system, delay=adc.delay
    )

    gx_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-gx_readout.area,
        channel='x',
        grad_start=gx_readout.last,
        grad_end=0,
        system=system,
    )

    gy_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-gy_readout.area,
        channel='y',
        grad_start=gy_readout.last,
        grad_end=0,
        system=system,
    )

    gx_readout_list.append(gx_readout)
    gy_readout_list.append(gy_readout)
    gx_rewinder_list.append(gx_rewinder)
    gy_rewinder_list.append(gy_rewinder)

    rewinder_duration = max(rewinder_duration, pp.calc_duration(gx_rewinder, gy_rewinder))


# gradient spoiling
A_gz_spoil = 4 / slice_thickness - gz.area / 2
gz_spoil = pp.make_trapezoid(channel='z', area=A_gz_spoil, system=system)

# update rewinder duration
rewinder_duration = max(rewinder_duration, pp.calc_duration(gz_spoil))

# calculate minimum echo time (TE) for sequence header
min_TE = pp.calc_duration(gz) / 2 + pp.calc_duration(gzr)
min_TE = np.ceil(min_TE / system.grad_raster_time) * system.grad_raster_time  # put on raster
min_TE = np.ceil(min_TE * 1e9) / 1e9  # round to 2 decimal values in ms

# calculate minimum repetition time (TR)
min_TR = (
    pp.calc_duration(gz)  # rf pulse
    + pp.calc_duration(gzr)  # slice selection re-phasing gradient
    + pp.calc_duration(gx_readout_list[0], adc)  # readout
    + rewinder_duration  # max of rewinder gradients / gz_spoil durations
    + time_label
)

min_TR = np.ceil(min_TR / system.grad_raster_time) * system.grad_raster_time  # put on raster
min_TR = np.ceil(min_TR * 1e9) / 1e9  # round to 2 decimal values in ms

# calculate TR delay
tr_delay = (
    time_label
    if TR is None
    else np.ceil((TR - min_TR + time_label) / system.grad_raster_time) * system.grad_raster_time
)
assert tr_delay >= 1e-5, f'TR must be larger than {min_TR * 1000:.2f} ms. Current value is {TR * 1000:.2f} ms.'

# print TR values
final_TR = min_TR if TR is None else min_TR + tr_delay - time_label
print(f'\n shortest TR = {min_TR * 1000:.2f} ms')
print(f'\n final TR = {final_TR * 1000:.2f} ms')


# # # # # # # # # # # # #
# CREATE ISMRMRD HEADER #
# # # # # # # # # # # # #

# define full filename
str_sampling = 'golden_angle' if FLAG_GOLDEN_ANGLE else 'uniform'
str_vds = '_vds' if fov_coeff[1] != 0 else ''
filename += f'_{adc_total_samples}k0_{n_spirals}k1_{str_sampling}{str_vds}'

# create folder for seq and header file
output_path = Path.cwd() / 'output' / filename
output_path.mkdir(parents=True, exist_ok=True)

# delete existing header file
if (output_path / f'{filename}_header.h5').exists():
    (output_path / f'{filename}_header.h5').unlink()

# create header
#hdr = create_hdr(
#    traj_type='spiral',
#    fov=fov,
#    res=res,
#    slice_thickness=slice_thickness,
#    dt=adc_dwell,
#    n_k1=n_spirals,
#)

# write header to file
prot = ismrmrd.Dataset(output_path / f'{filename}_header.h5', 'w')
prot.write_xml_header(hdr.toXML('utf-8'))

# choose initial rf phase offset
rf_phase = 0
rf_inc = 0

# # # # # # # # # # # # # # # # #
# ADD BLOCKS TO SEQUENCE OBJECT #
# # # # # # # # # # # # # # # # #

# initiate LIN label. Might not be required.
seq.add_block(pp.make_delay(time_label), pp.make_label(label='LIN', type='SET', value=0))

for idx in range(n_spirals):
    # set current phase_offset if rf_spoiling is activated
    if rf_spoiling_inc > 0:
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi

    # add slice selective excitation pulse
    seq.add_block(rf, gz)

    # add slice selection re-phasing gradient
    seq.add_block(gzr)

    # add readout gradients and ADC
    seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)

    # add rewinder gradients and spoiler
    gx_rewinder = gx_rewinder_list[idx]
    gy_rewinder = gy_rewinder_list[idx]
    seq.add_block(gx_rewinder, gy_rewinder, gz_spoil)

    # calculate rewinder delay for current shot
    current_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
    rewinder_delay = rewinder_duration - current_rewinder_duration

    # add delay to ensure constant TR and increment LIN label
    seq.add_block(pp.make_delay(rewinder_delay + tr_delay), pp.make_label(label='LIN', type='INC', value=1))

    # add acquisitions to metadata
    acq = ismrmrd.Acquisition()
    acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
    traj_ismrmrd = np.stack([spiral_trajectory[idx, 0, 0:-1] * fov, spiral_trajectory[idx, 1, 0:-1] * fov]).T
    acq.traj[:] = traj_ismrmrd
    prot.append_acquisition(acq)

    # update rf phase offset for the next shot
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

# close ISMRMRD file
prot.close()

# check timing of the sequence
if FLAG_TIMINGCHECK and not FLAG_TESTREPORT:
    ok, error_report = seq.check_timing()
    if ok:
        print('\nTiming check passed successfully')
    else:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)

# show advanced rest report
if FLAG_TESTREPORT:
    print('\nCreating advanced test report...')
    print(seq.test_report())

# write all required parameters in the seq-file definitions.
# Number of MRF repetitions need to be passed as 'Ny' and will be saved as 'k_space_encoding1'.
tr_value = TR if TR is not None else min_TR
write_seq_definitions(
    seq=seq,
    fov=fov,
    slice_thickness=slice_thickness,
    name=filename,
    alpha=rf_angle,
    Nx=adc_total_samples,
    Ny=n_spirals,
    sampling_scheme='spiral',
    N_slices=1,
    TR=tr_value,
    TE=min_TE,
    delta=delta_angle,
)

# save seq-file
if FLAG_WRITE_SEQ:
    print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
    seq.write(str(output_path / filename), create_signature=True)


if FLAG_PLOTS:
    seq.plot()

    # calculate k-space trajectory from sequence
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # plot k-space trajectory
    plt.figure()
    plt.plot(k_traj[0], k_traj[1], 'b')
    plt.plot(k_traj_adc[0], k_traj_adc[1], '.', color='red', markersize=3)
    plt.show()