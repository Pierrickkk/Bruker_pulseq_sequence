import math

import numpy as np

import pypulseq as pp

FLAG_PLOT = False
FLAG_WRITE_SEQ = True
seq_filename = "RF_test_adia_seul.seq"
# ======
# SETUP
# ======
# Create a new sequence object
fov = 60e-3  # Define FOV and resolution
Nx = 64
Ny = 64
alpha = 10  # flip angle
slice_thickness = 3e-3  # slice
TR = 12e-3  # Repetition time
TE = 5e-3  # Echo time
n_slices = 1
rf_spoiling_inc = 117  # RF spoiling increment

# Set system limits
system = pp.Opts(
    max_grad=300,
    grad_unit='mT/m',
    max_slew=2000,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100.5e-6,
    adc_dead_time=100e-6,
    grad_raster_time=10e-6, # should be 8 ?
)

seq = pp.Sequence(system)

# ======
# CREATE EVENTS
# ======
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=alpha * math.pi / 180,
    duration=3e-3,
    slice_thickness=slice_thickness,
    #apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    delay=system.rf_dead_time,
)
rf2 = pp.make_sinc_pulse(
    flip_angle=(alpha+10) * math.pi / 180,
    duration=15e-3,
    slice_thickness=slice_thickness,
    apodization=0.42,
    time_bw_product=4,
    system=system,
    delay=system.rf_dead_time,
)
rf_adiabatic = pp.make_adiabatic_pulse(
    adiabaticity=4,
    pulse_type='wurst',
    duration=5e-3,
    slice_thickness=slice_thickness,
    #apodization=0.42,
    use='undefined',
    system=system,
    delay=system.rf_dead_time,
    )

# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=3.2e-3, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k


# gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=2 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = (
    np.ceil(
        (TE - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2)
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)
delay_TR = (
    np.ceil(
        (TR - pp.calc_duration(gz) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - delay_TE)
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)

assert np.all(delay_TE >= 0)
assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0





# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over phase encodes and define sequence blocks
for i in range(1):
    
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    seq.add_block(rf_adiabatic, gz) 
    seq.add_block(pp.make_delay(0.1e-3))
    #seq.add_block(rf, gz)
    seq.add_block(pp.make_delay(0.1e-3))
    gy_pre = pp.make_trapezoid(
        channel='y',
        area=phase_areas[i],
        duration=pp.calc_duration(gx_pre),
        system=system,
    )
    
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(pp.make_delay(delay_TE))
    seq.add_block(gx, adc)
    gy_pre.amplitude = -gy_pre.amplitude
    seq.add_block(pp.make_delay(delay_TR), gx_spoil, gy_pre, gz_spoil)

# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# ======
# VISUALIZATION
# ======
if FLAG_PLOT:
    seq.plot()

seq.calculate_kspace()

# Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
# slew-rate limits
rep = seq.test_report()
print(rep)

# =========
# WRITE .SEQ
# =========
if FLAG_WRITE_SEQ:
    # Prepare the sequence output for the scanner
    seq.set_definition(key='system_max_grad', value=system.max_grad)
    seq.set_definition(key='system_max_slew', value=system.max_slew)
    seq.set_definition(key='system_rf_ringdown_time', value=system.rf_ringdown_time)
    seq.set_definition(key='system_rf_dead_time', value=system.rf_dead_time)
    seq.set_definition(key='AdcDeadTime', value=system.adc_dead_time)
    seq.set_definition(key='system_grad_raster_time', value=system.grad_raster_time)

    # geometry
    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Matrix', value=[Nx, Ny, 1])
    seq.set_definition(key='nslices', value=n_slices)

    # contrast
    seq.set_definition(key='alpha', value=alpha)
    seq.set_definition(key='TE', value=TE)
    seq.set_definition(key='TR', value=TR)

    # remaining seq parameters
    seq.set_definition(key='rf_spoiling_inc', value=rf_spoiling_inc)
    #seq.set_definition(key='ro_duration', value=ro_duration)
    #seq.set_definition(key='spoiler_duration', value=spoiler_duration)


    seq.write("/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/python/Test_unitaire/RF_intensities/output" +"/"+ seq_filename)




