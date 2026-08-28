import math

import numpy as np
from scipy.interpolate import interp1d
import pypulseq as pp

FLAG_PLOT = True
FLAG_WRITE_SEQ = True

seq_filename = "shape7_rf.seq"

filename_sech_inv = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/shape/sech.inv"
filename_sech_exc = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/shape/sech.exc"
filename_sinc_exc = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/shape/sinc.exc"
filename_sinc_inv = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/shape/sinc.inv"

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
    rf_dead_time=100e-6,
    adc_dead_time=100e-6,
    grad_raster_time=10e-6, # should be 8 ?
)

seq = pp.Sequence(system)

def load_and_stretch_rf(filename: str, rf_raster_time: float, new_duration: float = None) -> tuple:
    """
    Parse a Bruker .exc file and optionally time-stretch the RF pulse.

    Parameters
    ----------
    filename : str
        Path to the .exc file.
    rf_raster_time : float
        Dwell time per sample in seconds (e.g. system.rf_raster_time).
    new_duration : float, optional
        Target pulse duration in seconds. Prompts the user if None.

    Returns
    -------
    rf_complex : np.ndarray
        Original complex RF signal (mag * exp(j*phase)).
    rf_complex_stretched : np.ndarray
        Time-stretched complex RF signal resampled to new_duration.
    new_duration : float
        The target duration actually used.
    """
    mag, phase_rad = [], []

    with open(filename, "r") as f:
        start = False
        for line in f:
            line = line.strip()
            if "##XYPOINTS=" in line:
                start = True
                continue
            if not start or not line:
                continue
            try:
                xi, yi = line.split(",")
                mag.append(float(xi))
                phase_rad.append(np.deg2rad(float(yi)))
            except ValueError:
                break  # end of data block

    rf_complex = np.array(mag) * np.exp(1j * np.array(phase_rad))

    if new_duration is None:
        old_duration = len(rf_complex) * rf_raster_time
        new_duration = float(input(
            f"Original duration: {old_duration * 1e3:.3f} ms ? enter new duration (ms): "
        )) * 1e-3

    n_new = int(round(len(rf_complex) * new_duration / (len(rf_complex) * rf_raster_time)))
    x_old = np.linspace(0, 1, len(rf_complex))
    x_new = np.linspace(0, 1, n_new)

    rf_complex_stretched = (
        interp1d(x_old, np.real(rf_complex), kind="cubic")(x_new)
        + 1j * interp1d(x_old, np.imag(rf_complex), kind="cubic")(x_new)
    )

    return rf_complex, rf_complex_stretched, new_duration


def read_bruker_header(filename):
    params = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            for key in ["SHAPE_INTEGFAC", "SHAPE_BWFAC", "SHAPE_TOTROT"]:
                if line.startswith(f"##${key}="):
                    params[key] = float(line.split("=")[1])
    return params

# Bruker hard pulse reference: 90° in 1ms
# On Bruker systems, the reference B1 for a 1ms hard 90° pulse
# is typically calibrated and stored. In pypulseq units (Hz = cycles/s):
gamma_hz_per_T = 42.577e6  # Hz/T

# Reference: hard pulse 90° in 1ms ? B1_ref in Tesla
flip_ref = np.pi / 2          # 90°
dur_ref  = 1e-3               # 1 ms
B1_ref   = flip_ref / (2 * np.pi * gamma_hz_per_T * dur_ref)
# B1_ref ? 5.87e-6 T  ? 5.87 µT ? 250 Hz in flip-angle-rate units
def compute_bruker_peak_b1(flip_angle_rad, duration_s, integfac, B1_ref, dur_ref=1e-3):
    """
    Bruker scaling formula:
      B1_peak = B1_ref × (flip/90°) × (dur_ref/duration) × (1/integfac)
    
    integfac : SHAPE_INTEGFAC from the .exc header (normalized integral)
    """
    return B1_ref * (flip_angle_rad / (np.pi/2)) * (dur_ref / duration_s) * (1.0 / integfac)

def scale_bruker_rf(signal_complex, flip_angle_rad, duration_s, integfac):
    """
    Returns the signal array scaled to match Bruker amplitude,
    ready for make_arbitrary_rf with no_signal_scaling=True.
    """
    gamma_hz_per_T = 42.577e6
    dur_ref  = 1e-3
    flip_ref = np.pi / 2
    B1_ref   = flip_ref / (2 * np.pi * gamma_hz_per_T * dur_ref)  # ~5.87e-6 T

    B1_peak  = compute_bruker_peak_b1(flip_angle_rad, duration_s, integfac, B1_ref)
    B1_peak_hz = B1_peak * gamma_hz_per_T  # convert T ? Hz

    # signal is normalized: max(|signal|) = 1
    signal_norm = signal_complex / np.max(np.abs(signal_complex))
    return signal_norm * B1_peak_hz






rf_complex, rf_stretched, dur = load_and_stretch_rf(
    filename_sech_exc,
    rf_raster_time=system.rf_raster_time,
    new_duration=6e-3   # pass directly, or omit to be prompted
)

sinc_complex, sinc_stretched, dur = load_and_stretch_rf(
    filename_sinc_exc,
    rf_raster_time=system.rf_raster_time,
    new_duration=6e-3   # pass directly, or omit to be prompted
)




header = read_bruker_header(filename_sinc_exc)
integfac = header["SHAPE_INTEGFAC"]   # e.g. 0.4637 for a sinc

flip = 90 * np.pi / 180
dur  = 3e-6 * len(sinc_complex)       # original duration (dwell × n_points)

signal_scaled = scale_bruker_rf(sinc_complex, flip, dur, integfac)

sinc_bruk = pp.make_arbitrary_rf(
    no_signal_scaling=False,
    signal=signal_scaled,
    dwell=3e-6,
    flip_angle=flip,          # ignoré par pypulseq quand no_signal_scaling=True
    delay=system.rf_dead_time
)

# ======
# CREATE EVENTS
# ======
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=90 * math.pi / 180,
    duration=3e-3,
    slice_thickness=slice_thickness,
    #apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    delay=system.rf_dead_time,
)

sinc_bruk_inv = pp.make_arbitrary_rf(
    no_signal_scaling=True,
    signal=sinc_complex,
    dwell=3e-6,
    flip_angle=90* math.pi / 180,
    delay=system.rf_dead_time
)

sinc_bruk_inv.signal = sinc_bruk_inv.signal * (np.pi/2)/(2*np.pi*0.16895*sinc_bruk_inv.shape_dur *2e-6*100)

sinc_stretched_bruk_inv = pp.make_arbitrary_rf(
    no_signal_scaling=False,
    signal=sinc_stretched,
    flip_angle=10* math.pi / 180,
    delay=system.rf_dead_time
)

rf1_bruk_inv = pp.make_arbitrary_rf(
    no_signal_scaling=True,
    signal=rf_complex,
    dwell=3e-6,
    flip_angle=90* math.pi / 180,
    delay=system.rf_dead_time

)
rf1_bruk_inv.signal = rf1_bruk_inv.signal * (2.50 * (1e-3/6e-3) * 0.1064278)

rf2_bruk_inv = pp.make_arbitrary_rf(
    no_signal_scaling=True,
    signal=rf_complex,
    flip_angle=90* math.pi / 180,
    delay=system.rf_dead_time
)

rf2_bruk_inv.signal = rf2_bruk_inv.signal * (250 * (1e-3/6e-3) * 0.1064278)  # scale the signal to match the peak of the bruker sech.inv pulse; note the sqrt(1000) from above.


rf12_bruk_stretched_inv = pp.make_arbitrary_rf(
    no_signal_scaling=False,
    signal=rf_stretched,
    flip_angle=90* math.pi / 180,
    delay=system.rf_dead_time

)

hardpulserefduration = 1e-3
pulseduration = 6e-3
hard_peak_b1 = 83 * (hardpulserefduration / pulseduration)   # in microTesla, to match the peak of the bruker sech.inv pulse; note the sqrt(1000) from above.
rf_inv_dur = 6.0e-3   #6 ms inversion sech pulse
mu = 5   #found by iterating to get the same amplitude integral (measured integral, not the one stated in the pulse file).
beta_inv = 1760
K_inv = (2*np.pi * hard_peak_b1*np.sqrt(1000))**2 / (mu * beta_inv**2)   # finds the K to achieve the same B1 peak as the bruker sech.inv pulse; note the sqrt(1000) from above. 
#K is about 17.7 for the sech.inv pulse

rf_inv_MB = pp.make_adiabatic_pulse(
    pulse_type='hypsec',
    duration=rf_inv_dur,
    adiabaticity=K_inv,
    mu=mu,
    beta=beta_inv,
    system=system,
    return_gz=False,
    delay=system.rf_dead_time,
)

## Define other gradients and ADC events
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


    seq.add_block(sinc_bruk, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(rf, gz) 
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(sinc_bruk_inv, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(sinc_stretched_bruk_inv, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(rf2_bruk_inv, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(rf1_bruk_inv, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(rf12_bruk_stretched_inv, gz,pp.make_delay(7e-3))
    seq.add_block(pp.make_delay(0.1e-3))
    seq.add_block(rf_inv_MB, gz,pp.make_delay(7e-3))
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




