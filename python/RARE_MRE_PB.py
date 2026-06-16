import warnings
import math
import time
import copy
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

FLAG_MRE          = True
FLAG_TRIG         = True
FLAG_MRE_BIPOLAR  = True #false for unipolar meg



# ======
# SEQUENCE PARAMETERS
# ======
fov = 60e-3
Nx = 128
Ny = 128

n_echo = 1

n_slices = 1

rf_flip_deg = 180

slice_thickness = 1e-3

TE = 20e-3
TR = 1000e-3


output_path = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/output"


# ======
# MRE PARAMETERS
# ======
mre_exc_freq       = 500.0        # single mechanical excitation frequency [Hz]
mre_wave_period    = 1 / mre_exc_freq
mre_n_timesteps    = 1            # number of phase offsets (time steps) over one wave period
mre_meg_cycles     = 3           # number of MEG cycles (bipolar gradient pairs)
mre_meg_orientations =  ['y']        #['x', 'y', 'z']
mre_exp_number     = 10            # experiment number encoded in trigger pulse width


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


# =============
# RF PARAMETERS
# =============
if isinstance(rf_flip_deg, int):
    rf_flip_deg = np.zeros(n_echo) + rf_flip_deg

sampling_time = 6.4e-3
readout_time = sampling_time + 2 * system.adc_dead_time

if FLAG_DWELL_BRUKER:
    dwell_time = 50e-6
    readout_time = (dwell_time * Nx)+ 2 * system.adc_dead_time # ADC duration
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
        print("adc dead time now")
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
# Motion encoding gradients (zeroth-order moment nulling: +/-)
# Each MEG cycle = one positive lobe + one negative lobe of duration mre_meg_duration/2
meg_lobe_dur = mre_wave_period / 2
meg_lobe_plus = pp.make_trapezoid(channel='x', amplitude=abs(system.max_grad / 2),
                          duration=meg_lobe_dur, system=system)
# ================
# Meg construction
# ================
amp = abs(system.max_grad / 2)

rt = meg_lobe_plus.rise_time
ft = meg_lobe_plus.flat_time
tt = rt + ft + meg_lobe_plus.fall_time

times = [0]
amps = [0]

current_t = 0

for i in range(mre_meg_cycles):


    # ramp up
    current_t += rt
    times.append(current_t)
    amps.append(amp)

    # flat
    current_t += ft
    times.append(current_t)
    amps.append(amp)

    # ramp down
    current_t += rt
    times.append(current_t)
    amps.append(0)

    # ramp up
    current_t += rt
    times.append(current_t)
    amps.append(-amp)

    # flat
    current_t += ft
    times.append(current_t)
    amps.append(-amp)

    # ramp down
    current_t += rt
    times.append(current_t)
    amps.append(0)

meg = pp.make_extended_trapezoid(
    channel='x',
    times=np.array(times),
    amplitudes=np.array(amps),
    system=system,
) 

if FLAG_MRE:
    total_meg_dur = pp.calc_duration(meg)
else:
    total_meg_dur = 0

trig_out = pp.make_digital_output_pulse('osc1', duration=100e-6, delay=0)

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

te_train = t_ex_calc + n_echo * t_ref_calc + t_end + total_meg_dur
te_train = (
    system.grad_raster_time
    * np.round(te_train / system.grad_raster_time)
)
tr_delay = (TR - n_slices * te_train - pp.calc_duration(trig_out)*FLAG_TRIG) / n_slices

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


if FLAG_TRIG:
    assert np.all(math.ceil(te_train - trig_out.duration))
    min_TR = math.ceil((tr_delay - trig_out.duration)/seq.grad_raster_time)*seq.grad_raster_time
if FLAG_MRE:
    tr_delay = np.ceil(tr_delay / mre_wave_period) * mre_wave_period  # ensure TR is a multiple of the wave period
    assert np.all((tr_delay // mre_wave_period) >= 1)

# ======
# CONSTRUCT SEQUENCE
# ======
for n_dim, meg_orientation in enumerate(mre_meg_orientations):
    #seq.add_block(pp.make_delay(time_label),
                 # make_label(type='SET', label='SET', value=n_dim))

    meg.channel = meg_orientation

    for idx_timesteps in range(mre_n_timesteps):

        delay_timestep = idx_timesteps / mre_n_timesteps * mre_wave_period
        #seq.add_block(pp.make_delay(delay_timestep))

        # Loop over slices
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
                if FLAG_TRIG:
                    seq.add_block(trig_out)
                if delay_timestep>0:
                        seq.add_block(pp.make_delay(delay_timestep))

                # Excitation
                seq.add_block(gs1)
                seq.add_block(rf_ex, gs2)
                if FLAG_MRE:                    
                    seq.add_block(gs3, meg, gr3)
                else:
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
                if FLAG_MRE_BIPOLAR:
                    if FLAG_TRIG:
                        seq.add_block(trig_out)
                # Excitation
                if delay_timestep>0:
                        seq.add_block(pp.make_delay(delay_timestep))
                seq.add_block(gs1)
                seq.add_block(rf_ex, gs2)
                if FLAG_MRE:
                    seq.add_block(gs3,meg, gr3)

                else:
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
seq.set_definition(key='FLAG_SHOW_PLOTS', value=int(FLAG_SHOW_PLOTS))
seq.set_definition(key='FLAG_TEST_REPORT', value=int(FLAG_TEST_REPORT))
seq.set_definition(key='FLAG_WRITE_SEQ', value=int(FLAG_WRITE_SEQ))

seq.set_definition(key='FLAG_DWELL_BRUKER', value=int(FLAG_DWELL_BRUKER))

seq.set_definition(key='FLAG_MRE', value=int(FLAG_MRE))
seq.set_definition(key='FLAG_TRIG', value=int(FLAG_TRIG))
seq.set_definition(key='FLAG_MRE_BIPOLAR', value=int(FLAG_MRE_BIPOLAR))


seq.set_definition(key='FLAG_SHOW_PLOTS', value=int(FLAG_SHOW_PLOTS))
seq.set_definition(key='FLAG_TEST_REPORT', value=int(FLAG_TEST_REPORT))
seq.set_definition(key='FLAG_WRITE_SEQ', value=int(FLAG_WRITE_SEQ))

seq.set_definition(key='FLAG_DWELL_BRUKER', value=int(FLAG_DWELL_BRUKER))

seq.set_definition(key='FLAG_MRE', value=int(FLAG_MRE))
seq.set_definition(key='FLAG_TRIG', value=int(FLAG_TRIG))
seq.set_definition(key='FLAG_MRE_BIPOLAR', value=int(FLAG_MRE_BIPOLAR))

seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
seq.set_definition(key='Name', value='tse')

seq.set_definition(key='Matrix', value=[Nx, Ny, n_slices])

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

    seq.plot(time_range=[0, 5 * TR])

    k_traj_adc, k_traj, *_ = seq.calculate_kspace()

    plt.figure()

    N3 = 128*2
    N2 = 30
    plt.plot(k_traj[0, 1:N3 ],
             k_traj[1, 1:N3 ],
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
        f"1506_RARE"
        f"_{Nx}"
        f"_{int(fov * 1e3)}mm"
        f"_ETL{n_echo}"
    )
    if FLAG_MRE:
        filename += (
            f"_{int(mre_exc_freq)}"
            f"_{mre_n_timesteps}ts"
            f"_{mre_meg_cycles}cyc"
        )

    print(filename)

    seq.write(output_path + "/" + filename + ".seq")