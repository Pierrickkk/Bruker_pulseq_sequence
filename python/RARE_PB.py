import warnings
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp


# ======
# FLAGS
# ======
FLAG_SHOW_PLOTS   = False
FLAG_TEST_REPORT  = True
FLAG_WRITE_SEQ    = True

FLAG_DWELL_BRUKER = True # True for dwell bruker friendly 



# ======
# SEQUENCE PARAMETERS
# ======
fov = 30e-3
Nx = 128
Ny = 128

n_echo = 8
n_slices = 1

rf_flip_deg = 180

slice_thickness = 0.3e-3

TE = 20e-3
TR = 1000e-3


output_path = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/output"



# ======
# SYSTEM LIMITS
# ======
system = pp.Opts(
    max_grad=300,
    grad_unit='mT/m',
    max_slew=2000,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=100e-6,
)


seq = pp.Sequence(system)



dG = 250e-6


# ======
# RF PARAMETERS
# ======
if isinstance(rf_flip_deg, int):
    rf_flip_deg = np.zeros(n_echo) + rf_flip_deg

sampling_time = 6.4e-3
readout_time = sampling_time + 2 * system.adc_dead_time

if FLAG_DWELL_BRUKER:
    dwell_time = 50e-6
    readout_time = dwell_time * Nx # ADC duration
    readout_time=round(readout_time/system.block_duration_raster)*system.block_duration_raster

t_ex = 2.5e-3
t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time

t_ref = 2e-3
t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time

t_sp = 0.5 * (TE - readout_time - t_refwd)
t_spex = 0.5 * (TE - t_exwd - t_refwd)

fsp_r = 1
fsp_s = 0.5

rf_ex_phase = np.pi / 2
rf_ref_phase = 0


# ======
# EXCITATION PULSE
# ======
flip_ex = np.deg2rad(90)

rf_ex, gz, _ = pp.make_sinc_pulse(
    flip_angle=flip_ex,
    system=system,
    duration=t_ex,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    phase_offset=rf_ex_phase,
    return_gz=True,
    delay=system.rf_dead_time,
    use='excitation',
)

gs_ex = pp.make_trapezoid(
    channel='z',
    system=system,
    amplitude=gz.amplitude,
    flat_time=t_exwd,
    rise_time=dG,
)


# ======
# REFOCUSING PULSE
# ======
flip_ref = np.deg2rad(rf_flip_deg[0])

rf_ref, gz, _ = pp.make_sinc_pulse(
    flip_angle=flip_ref,
    system=system,
    duration=t_ref,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    phase_offset=rf_ref_phase,
    use='refocusing',
    return_gz=True,
    delay=system.rf_dead_time,
)

gs_ref = pp.make_trapezoid(
    channel='z',
    system=system,
    amplitude=gs_ex.amplitude,
    flat_time=t_refwd,
    rise_time=dG,
)


# ======
# SLICE GRADIENTS
# ======
ags_ex = gs_ex.area / 2

gs_spr = pp.make_trapezoid(
    channel='z',
    system=system,
    area=ags_ex * (1 + fsp_s),
    duration=t_sp,
    rise_time=dG,
)

gs_spex = pp.make_trapezoid(
    channel='z',
    system=system,
    area=ags_ex * fsp_s,
    duration=t_spex,
    rise_time=dG,
)


# ======
# K-SPACE PARAMETERS
# ======
delta_kx = 1 / fov
delta_ky = 1 / fov

k_width = Nx * delta_kx


# ======
# READOUT GRADIENTS
# ======
gr_acq = pp.make_trapezoid(
    channel='x',
    system=system,
    flat_area=k_width,
    flat_time=readout_time,
    rise_time=dG,
)
if FLAG_DWELL_BRUKER:
    adc_delay= gr_acq.rise_time + ( gr_acq.flat_time- readout_time)/2
    adc_delay = round(adc_delay/system.block_duration_raster)*system.block_duration_raster
    if (adc_delay<system.adc_dead_time):
        gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_kx, flat_time=readout_time, rise_time=system.adc_dead_time,system=system)
    adc = pp.make_adc(num_samples=Nx, dwell=dwell_time, delay=adc_delay, system=system)
else:
    adc = pp.make_adc(num_samples=Nx, duration=sampling_time, delay=system.adc_dead_time, system=system)


gr_spr = pp.make_trapezoid(
    channel='x',
    system=system,
    area=gr_acq.area * fsp_r,
    duration=t_sp,
    rise_time=dG,
)

agr_spr = gr_spr.area
agr_preph = gr_acq.area / 2 + agr_spr

gr_preph = pp.make_trapezoid(
    channel='x',
    system=system,
    area=agr_preph,
    duration=t_spex,
    rise_time=dG,
)


# ======
# PHASE ENCODING
# ======
n_ex = int(np.floor(Ny / n_echo))

pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1

if divmod(n_echo, 2)[1] == 0:
    pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])

pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

phase_areas = pe_order * delta_ky


# ======
# EXTENDED GRADIENTS
# ======
gs1 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([0, gs_ex.rise_time]),
    amplitudes=np.array([0, gs_ex.amplitude]),
    system=system,
)

gs2 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([0, gs_ex.flat_time]),
    amplitudes=np.array([gs_ex.amplitude, gs_ex.amplitude]),
    system=system,
)

gs3 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([
        0,
        gs_spex.rise_time,
        gs_spex.rise_time + gs_spex.flat_time,
        gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time,
    ]),
    amplitudes=np.array([
        gs_ex.amplitude,
        gs_spex.amplitude,
        gs_spex.amplitude,
        gs_ref.amplitude,
    ]),
    system=system,
)

gs4 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([0, gs_ref.flat_time]),
    amplitudes=np.array([gs_ref.amplitude, gs_ref.amplitude]),
    system=system,
)

gs5 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([
        0,
        gs_spr.rise_time,
        gs_spr.rise_time + gs_spr.flat_time,
        gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
    ]),
    amplitudes=np.array([
        gs_ref.amplitude,
        gs_spr.amplitude,
        gs_spr.amplitude,
        0,
    ]),
    system=system,
)

gs7 = pp.make_extended_trapezoid(
    channel='z',
    times=np.array([
        0,
        gs_spr.rise_time,
        gs_spr.rise_time + gs_spr.flat_time,
        gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
    ]),
    amplitudes=np.array([
        0,
        gs_spr.amplitude,
        gs_spr.amplitude,
        gs_ref.amplitude,
    ]),
    system=system,
)


# ======
# READOUT EXTENDED GRADIENTS
# ======
gr3 = gr_preph

gr5 = pp.make_extended_trapezoid(
    channel='x',
    times=np.array([
        0,
        gr_spr.rise_time,
        gr_spr.rise_time + gr_spr.flat_time,
        gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
    ]),
    amplitudes=np.array([
        0,
        gr_spr.amplitude,
        gr_spr.amplitude,
        gr_acq.amplitude,
    ]),
    system=system,
)

gr6 = pp.make_extended_trapezoid(
    channel='x',
    times=np.array([0, readout_time]),
    amplitudes=np.array([gr_acq.amplitude, gr_acq.amplitude]),
    system=system,
)

gr7 = pp.make_extended_trapezoid(
    channel='x',
    times=np.array([
        0,
        gr_spr.rise_time,
        gr_spr.rise_time + gr_spr.flat_time,
        gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
    ]),
    amplitudes=np.array([
        gr_acq.amplitude,
        gr_spr.amplitude,
        gr_spr.amplitude,
        0,
    ]),
    system=system,
)


# ======
# TIMING
# ======
t_ex_calc = (
    pp.calc_duration(gs1)
    + pp.calc_duration(gs2)
    + pp.calc_duration(gs3)
)

t_ref_calc = (
    pp.calc_duration(gs4)
    + pp.calc_duration(gs5)
    + pp.calc_duration(gs7)
    + readout_time
)

t_end = (
    pp.calc_duration(gs4)
    + pp.calc_duration(gs5)
)

te_train = t_ex_calc + n_echo * t_ref_calc + t_end
te_train = (
    system.grad_raster_time
    * np.round(te_train / system.grad_raster_time)
)
tr_delay = (TR - n_slices * te_train) / n_slices

tr_delay = (
    system.grad_raster_time
    * np.round(tr_delay / system.grad_raster_time)
)

if tr_delay < 0:
    tr_delay = 1e-3
    warnings.warn(
        f'TR too short, adapted to: '
        f'{1000 * n_slices * (te_train + tr_delay)} ms'
    )
else:
    print(f'TR delay: {1000 * tr_delay} ms')


# ======
# CONSTRUCT SEQUENCE
# ======
for i_excitation in range(n_ex + 1):

    for i_slice in range(n_slices):

        rf_ex.freq_offset = (
            gs_ex.amplitude
            * slice_thickness
            * (i_slice - (n_slices - 1) / 2)
        )

        rf_ref.freq_offset = (
            gs_ref.amplitude
            * slice_thickness
            * (i_slice - (n_slices - 1) / 2)
        )

        rf_ex.phase_offset = (
            rf_ex_phase
            - 2 * np.pi * rf_ex.freq_offset * pp.calc_rf_center(rf_ex)[0]
        )

        rf_ref.phase_offset = (
            rf_ref_phase
            - 2 * np.pi * rf_ref.freq_offset * pp.calc_rf_center(rf_ref)[0]
        )

        # Excitation
        seq.add_block(gs1)
        seq.add_block(rf_ex, gs2)
        seq.add_block(gs3, gr3)

        # Echo train
        for i_echo in range(n_echo):

            if i_excitation > 0:
                phase_area = phase_areas[i_echo, i_excitation - 1]
            else:
                phase_area = 0.0

            gp_pre = pp.make_trapezoid(
                channel='y',
                system=system,
                area=phase_area,
                duration=t_sp,
                rise_time=dG,
            )

            gp_rew = pp.make_trapezoid(
                channel='y',
                system=system,
                area=-phase_area,
                duration=t_sp,
                rise_time=dG,
            )

            seq.add_block(rf_ref, gs4)
            seq.add_block(gr5, gp_pre, gs5)

            if i_excitation > 0:
                seq.add_block(gr6, adc)
            else:
                seq.add_block(gr6)

            seq.add_block(gr7, gp_rew, gs7)

        seq.add_block(gs4)
        seq.add_block(gs5)

        seq.add_block(pp.make_delay(tr_delay))


# ======
# TIMING CHECK
# ======
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]


# ======
# TEST REPORT
# ======
if FLAG_TEST_REPORT:
    print(seq.test_report())


# ======
# DEFINITIONS
# ======
seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
seq.set_definition(key='Name', value='tse')

seq.set_definition(key='matrix', value=[Nx, Ny, n_slices])

seq.set_definition(key='nslices', value=n_slices)
seq.set_definition(key='n_echo', value=n_echo)

seq.set_definition(key='rf_flip_deg', value=rf_flip_deg[0])

seq.set_definition(key='TE', value=TE)
seq.set_definition(key='TR', value=TR)

seq.set_definition(key='AdcDeadTime', value=system.adc_dead_time)


# ======
# PLOTS
# ======
if FLAG_SHOW_PLOTS:

    seq.plot(time_range=[0, 2 * TR])

    k_traj_adc, k_traj, *_ = seq.calculate_kspace()

    plt.figure()

    N3 = 64 * 34

    plt.plot(k_traj[0, 1:N3 * 10],
             k_traj[1, 1:N3 * 10],
             'b')

    plt.plot(k_traj_adc[0, 1:N3],
             k_traj_adc[1, 1:N3],
             '.r',
             markersize=3)

    plt.title('k-space trajectory')

    plt.show()


# ======
# WRITE SEQUENCE
# ======
if FLAG_WRITE_SEQ:

    filename = (
        f"2905_TSE"
        f"_{Nx}"
        f"_{int(fov * 1e3)}mm"
        f"_ETL{n_echo}"
        f"_TR{int(TR * 1e3)}"
        f"_TE{int(TE * 1e3)}"
    )

    print(filename)

    seq.write(output_path + "/" + filename + ".seq")