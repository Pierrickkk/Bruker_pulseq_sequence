import math

import numpy as np
from pathlib import Path

import pypulseq as pp

output_path = '/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq'

def main(plot: bool = False, write_seq: bool = False, seq_filename: str = 'gradient_ext_trap_alone_pypulseq.seq'):
    # ======
    # SETUP
    # ======
    # Create a new sequence object
    fov = 256e-3  # Define FOV and resolution
    Nx = 64
    Ny = 64

    slice_thickness = 3e-3  # slice
    TR = 12e-3  # Repetition time
    TE = 5e-3  # Echo time


    system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=150,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    # ======
    # CREATE EVENTS
    # ======

    # Define other gradients and ADC events
    delta_k = 1 / fov
    gx = pp.make_trapezoid(
        channel='x',
        flat_area=Nx * delta_k,
        flat_time=3.2e-3,
        system=system,
        rise_time=2.e-5,
        fall_time=2.e-5)
    
    gx_split_rise,gx_split_flat,_ = pp.split_gradient(gx)
    gx_sum = pp.add_gradients([gx_split_rise,gx_split_flat],
                           system=system)



    gx2_opt,_,_ = pp.make_extended_trapezoid_area(
    channel="x",
    grad_start = gx.amplitude,
    grad_end = 0,
    area=4* Nx * delta_k,
    system=system)

    gx2_opt.delay=gx_sum.shape_dur

    # combine gradients
    g_tot = pp.add_gradients([gx_sum,gx2_opt],system=system)
    
    # Calculate timing
    delay_TE = (
        np.ceil(
            (TE - pp.calc_duration(g_tot) / 2)
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )
    delay_TR = (
        np.ceil(
            (TR - pp.calc_duration(g_tot) - delay_TE)
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )

    assert np.all(delay_TE >= 0)
    assert np.all(delay_TR >= 0)

    rf_phase = 0
    rf_inc = 0

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over phase encodes and define sequence blocks
    for i in range(Ny):
    
        seq.add_block(pp.make_delay(delay_TE))
        seq.add_block(g_tot)

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
    if plot:
        seq.plot(time_range=(0,TR))

    seq.calculate_kspace()

    # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
    # slew-rate limits
    #rep = seq.test_report()
    #print(rep)

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
        seq.set_definition(key='Name', value='gre')

        seq.write(output_path+"/"+seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=False, write_seq=True)