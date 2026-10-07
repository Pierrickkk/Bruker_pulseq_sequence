import pypulseq as pp
import numpy as np
import math
import matplotlib.pyplot as plt


# ======
# FLAGS
# ======
FLAG_SHOW_PLOTS = True
FLAG_WRITE_SEQ  = True
FLAG_INTERLEAVED_ADC = False

# ======
# SEQUENCE PARAMETERS
# ======
fov2 = 25e-3
Nx2 = 128
Ny2 = Nx2

fov = 20e-3
Nx = 128
Ny = Nx

alpha = 90
slice_thickness = 1e-3
n_slices = 1

TE = 15e-3
TR = 1000e-3

rf_spoiling_inc = 117

ro_duration = 3.2e-3
spoiler_duration = 3e-3
dwell_time2 = 20e-6
dwell_time = 21e-6

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
    grad_raster_time=10e-6,
)
n=1

# ======
# SEQUENCE
# ======
seq = pp.Sequence(system)

if FLAG_INTERLEAVED_ADC:
    n=2
for i in range(n):
    if i == 1:
        fov = fov2
        Nx = Nx2
        Ny = Ny2
        dwell_time = dwell_time2

    # ======
    # BRUKER-FRIENDLY ADC DWELL
    # ======
    

    ro_duration = dwell_time * Nx
    ro_duration = round(
        ro_duration / system.block_duration_raster
    ) * system.block_duration_raster


  


    # ======
    # RF + SLICE SELECTION
    # ======
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=alpha * np.pi / 180,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
    )


    # ======
    # READOUT GRADIENT + ADC
    # ======
    delta_k = 1 / fov

    gx = pp.make_trapezoid(
        channel='x',
        flat_area=Nx * delta_k,
        flat_time=ro_duration,
        system=system
    )


    adc_delay = gx.rise_time + (gx.flat_time - ro_duration) / 2

    adc_delay = round(
        adc_delay / system.block_duration_raster
    ) * system.block_duration_raster


    if adc_delay < system.adc_dead_time:
        gx = pp.make_trapezoid(
            channel='x',
            flat_area=Nx * delta_k,
            flat_time=ro_duration,
            rise_time=system.adc_dead_time,
            system=system
        )

    adc = pp.make_adc(
        num_samples=Nx,
        dwell=dwell_time,
        delay=adc_delay,
        system=system
    )


    # ======
    # PREPHASING GRADIENTS
    # ======
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2,
        duration=1e-3,
        system=system
    )

    gz_reph = pp.make_trapezoid(
        channel='z',
        area=-gz.area / 2,
        duration=1e-3,
        system=system
    )

    gy_pre = pp.make_trapezoid(
        channel='y',
        area=Ny / 2 * delta_k,
        duration=pp.calc_duration(gx_pre),
        system=system,
    )


    # ======
    # SPOILING
    # ======
    gx_spoil = pp.make_trapezoid(
        channel='x',
        area=2 * Nx * delta_k,
        duration=spoiler_duration,
        system=system
    )

    gz_spoil = pp.make_trapezoid(
        channel='z',
        area=4 / slice_thickness,
        duration=spoiler_duration,
        system=system
    )


    # ======
    # TIMING
    # ======
    min_TE = (
        math.ceil(
            (
                TE
                - pp.calc_duration(gx_pre)
                - gz.fall_time
                - gz.flat_time / 2
                - pp.calc_duration(gx) / 2
            )
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )

    min_TR = (
        math.ceil(
            (
                TR
                - pp.calc_duration(gz)
                - pp.calc_duration(gx_pre)
                - pp.calc_duration(gx)
                - min_TE
                - np.maximum(
                    pp.calc_duration(gx_spoil),
                    pp.calc_duration(gz_spoil)
                )
            )
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )


    assert np.all(min_TE >= 0)
    assert np.all(
        min_TR >= pp.calc_duration(gx_spoil, gz_spoil)
    )


    # ======
    # DELAYS
    # ======
    b_delay_TR = pp.make_delay(min_TR)
    b_delay_TE = pp.make_delay(min_TE)


    # ======
    # SEQUENCE CONSTRUCTION
    # ======
    rf_phase = 0
    rf_inc = 0

    scale_area = np.linspace(-1, 1, Ny)


    for s in range(n_slices):

        rf.freq_offset = gz.amplitude * slice_thickness * (
            s - (n_slices - 1) / 2
        )

        for i in range(Ny):

            # RF spoiling
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

            rf_inc = divmod(
                rf_inc + rf_spoiling_inc,
                360.0
            )[1]

            rf_phase = divmod(
                rf_phase + rf_inc,
                360.0
            )[1]

            # Excitation
            seq.add_block(rf, gz)

            # Phase encoding + rephasing
            seq.add_block(
                gx_pre,
                pp.scale_grad(
                    grad=gy_pre,
                    scale=scale_area[i]
                ),
                gz_reph
            )

            # TE delay
            seq.add_block(b_delay_TE)

            # Readout
            seq.add_block(gx, adc)

            # Spoiling
            seq.add_block(
                gx_spoil,
                pp.scale_grad(
                    grad=gy_pre,
                    scale=-scale_area[i]
                ),
                gz_spoil
            )

            # TR delay
            seq.add_block(b_delay_TR)


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
# SEQUENCE DEFINITIONS
# ======

# System
seq.set_definition(
    key='system_max_grad',
    value=system.max_grad
)

seq.set_definition(
    key='system_max_slew',
    value=system.max_slew
)

seq.set_definition(
    key='system_rf_ringdown_time',
    value=system.rf_ringdown_time
)

seq.set_definition(
    key='system_rf_dead_time',
    value=system.rf_dead_time
)

seq.set_definition(
    key='AdcDeadTime',
    value=system.adc_dead_time
)

seq.set_definition(
    key='system_grad_raster_time',
    value=system.grad_raster_time
)


# Geometry
seq.set_definition(
    key='FOV',
    value=[fov, fov, slice_thickness * n_slices]
)

seq.set_definition(
    key='Matrix',
    value=[Nx, Ny, 1])

seq.set_definition(key='nslices',value=n_slices)


# Contrast
seq.set_definition(key='alpha',value=alpha)

seq.set_definition(key='TE',value=TE)

seq.set_definition(key='TR',value=TR)


# Remaining parameters
seq.set_definition(key='rf_spoiling_inc',value=rf_spoiling_inc)
seq.set_definition(key='ro_duration',value=ro_duration)
seq.set_definition(key='spoiler_duration',value=spoiler_duration)
seq.set_definition(key='Name', value='bruker_gre')


# ======
# PLOTS
# ======
if FLAG_SHOW_PLOTS:

    #seq.plot(label='lin',time_range=np.array([0, 3]) * TR,time_disp='ms',grad_disp='mT/m')

    #seq.plot(time_range=np.array([0, 0.02]))

    seq.plot()

    k_traj_adc, k_traj, *_ = seq.calculate_kspace()

    plt.figure()

    plt.plot(
        k_traj[0],
        k_traj[1],
        'b'
    )

    plt.plot(
        k_traj_adc[0, :],
        k_traj_adc[1, :],
        '.r',
        markersize=3
    )

    plt.title('k-space trajectory')
    plt.show()


# ======
# WRITE SEQUENCE
# ======
if FLAG_WRITE_SEQ:

    output_path = (
        "/workspace_QMRI/PROJECTS_DATA/"
        "2026_RECH_bruker_pulseq/"
        "pypulseq/Sequences/Test_unitaire/"
        "multiple_ADC/output"
    )

    filename = (
        f"GRE_{Nx}_not_interleaved_swap_0610"
    )

    print(filename)

    seq.write(
        output_path + "/" + filename
    )