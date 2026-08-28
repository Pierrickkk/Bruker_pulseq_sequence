# %%
import math
import copy
from datetime import datetime
from math import pi
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from types import SimpleNamespace


### Flag
FLAG_WRITE_SEQ = False
FLAG_PLOT = True


# %%
# set system limits
system = pp.Opts(
    max_grad=28,
    grad_unit="mT/m",
    max_slew=180, # 90% de 188 T/(m*s)
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    grad_raster_time=10e-6
)

# %%
version_sequence = "v1.2.1"
# --------------Paramètres important--------------
fov = np.array([256, 256, 256]) * 1e-3  # Define FOV (m)
N = 128 # size of the reconstruction matrix
alpha=10                     # flip angle
TR = 5e-3                      # TR
nProj= 2                       # number of radial spokes
reord_traj = "standard" # "standard" / "golden" / "segment"
SGPoints = 0
nSeg = 1# number of segment nProj/nSeg must be an int
slice_thickness = 3e-3
# --------------Paramètres divers--------------#
riseT = 0.3e-3 #second -> 300us
# attention le dwell time doit être un multiple de 100ns
dwell = 10e-6
nDummy=100                      # number of dummy scans
rf_len = 0.05e-3 # second 
rf_spoiling_inc = 117 # RF spoiling increment
ro_spoil = 3  # Additional k-max excursion for RO spoiling
ro_os = 1  # Readout oversampling

# %%
res = fov[0]/N
delta_k = 1 / fov

# --- variable calculé à partir des valeurs
# number of sampled in readout to reach effective resolution
Ns = fov[0]/(2*res) + riseT/(2*dwell)
ro_dur =  dwell * Ns # seconde  
Ns


# %% [markdown]
# # Create RF object

# %%
rf = pp.make_block_pulse(
    flip_angle=alpha*pi/180,
    duration=rf_len,
    system=system)
# Trapezoidal

rf_trap, gz_trap, _ = pp.make_sinc_pulse(
        flip_angle=alpha * math.pi / 180,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.42, ## check with and without apodization
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
    )


gx_trap = pp.make_trapezoid(channel='x', flat_area=N * delta_k[0], flat_time=3.2e-3, system=system)
adc_trap = pp.make_adc(num_samples=N, duration=gx_trap.flat_time, delay=gx_trap.rise_time, system=system)
gx_trap_pre = pp.make_trapezoid(channel='x', area=-gx_trap.area / 2, duration=1e-3, system=system)
gz_trap_reph = pp.make_trapezoid(channel='z', area=-gz_trap.area / 2, duration=1e-3, system=system)
phase_areas_trap = (np.arange(N) - N / 2) * delta_k[0]

# gradient spoiling
gx_trap_spoil = pp.make_trapezoid(channel='x', area=2 * N * delta_k[0], system=system)
gz_trap_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)
# %%

adc = pp.make_adc(
    num_samples=Ns+SGPoints,
    duration=(Ns+SGPoints)*dwell,
    delay=0.0,
    system=system,
)

# gradient

gro = pp.make_trapezoid(
    channel="x",
    amplitude=1/(fov[0]*dwell),
    rise_time = riseT,
    flat_time=dwell*Ns-riseT,
    system=system
)

# split at the end of adc
gro_split_rise,gro_split_flat,_ = pp.split_gradient(gro)
gro_sum = pp.add_gradients([gro_split_rise,gro_split_flat],
                           system=system)



# %%
# spoil
gspoil = pp.make_trapezoid(
    channel="x",
    area =  2 * N * delta_k[0],
    system=system
)
gspoil

g_spoil_opt,a,b = pp.make_extended_trapezoid_area(
  channel="x",
  grad_start = gro.amplitude,
  grad_end = 0,
  area=2 * N * delta_k[0],
  system=system)

slewRate_up1 = gro.amplitude/gro.rise_time
step1 = slewRate_up1*(system.grad_raster_time/2)
wave1 = np.arange(0,gro.amplitude+step1, step1)
flat1 = gro.flat_time/(system.grad_raster_time/2)

wave2 = np.ones(int(flat1))*gro.amplitude
wave12 = np.concatenate((wave1,wave2))

amplitude2 = b[1] - b[0]
rise_time2 = a[1] - a[0]
slewRate_up2 = amplitude2/rise_time2
wave3 = np.arange(b[0],b[1], slewRate_up2*(system.grad_raster_time/2))
wave13 = np.concatenate((wave12, wave3))
# %%
flat_time2 = a[2] - a[1]
flat2 = flat_time2/(system.grad_raster_time/2)
wave4 = np.ones(int(flat2))*b[2]
wave14 = np.concatenate((wave13, wave4))

#%%
amplitude3 = b[3] - b[2] #negatif
fall_time3 = a[3] - a[2]
slewRate_up3 = amplitude3/fall_time3 # negatif
step3 = slewRate_up3*(system.grad_raster_time/2)
wave5 = np.arange(b[2],0, step3)
wave5 = np.append(wave5,[0,0])
wave15_OS= np.concatenate((wave14, wave5))

print("wave15", len(wave15_OS))
g_spoil_opt.delay=gro_sum.shape_dur

# %%
# combine gradients
g_tot = pp.add_gradients([gro_sum,g_spoil_opt],system=system)
g_tot

g_arb1 = pp.make_arbitrary_grad('x', wave15_OS, system=system,first=0,last=0, oversampling=True)

slewRate_up1 = gro.amplitude/gro.rise_time
step1 = slewRate_up1*(system.grad_raster_time)
wave1 = np.arange(0,gro.amplitude+step1, step1)
flat1 = gro.flat_time/(system.grad_raster_time)

wave2 = np.ones(int(flat1))*gro.amplitude
wave12 = np.concatenate((wave1,wave2))

amplitude2 = b[1] - b[0]
rise_time2 = a[1] - a[0]
slewRate_up2 = amplitude2/rise_time2
wave3 = np.arange(b[0],b[1], slewRate_up2*(system.grad_raster_time))
wave13 = np.concatenate((wave12, wave3))
# %%
flat_time2 = a[2] - a[1]
flat2 = flat_time2/(system.grad_raster_time)
wave4 = np.ones(int(flat2))*b[2]
wave14 = np.concatenate((wave13, wave4))

#%%
amplitude3 = b[3] - b[2] #negatif
fall_time3 = a[3] - a[2]
slewRate_up3 = amplitude3/fall_time3 # negatif
step3 = slewRate_up3*(system.grad_raster_time)
wave5 = np.arange(b[2],0, step3)
wave5 = np.append(wave5,0)
wave15= np.concatenate((wave14, wave5))

g_arb0 = pp.make_arbitrary_grad('x', wave15, system=system,first=0,last=0, oversampling=False)
# %%
# add SG delay
g_tot.delay = SGPoints*dwell
g_arb1.delay = SGPoints*dwell
g_arb0.delay = SGPoints*dwell
# %%
# Delay_TR
delay_TR = TR - (pp.calc_duration(rf)+
                 pp.calc_duration(g_tot))





#%%
# # Write real sequence

# %%
g_arb1_y = copy.deepcopy(g_arb1)
g_arb1_z = copy.deepcopy(g_arb1)
g_arb1_y.channel='y'
g_arb1_z.channel='z'


g_arb0_y = copy.deepcopy(g_arb0)
g_arb0_z = copy.deepcopy(g_arb0)
g_arb0_y.channel='y'
g_arb0_z.channel='z'

g_tot_y = copy.deepcopy(g_tot)
g_tot_z = copy.deepcopy(g_tot)
g_tot_y.channel='y'
g_tot_z.channel='z'


# %%
seq=pp.Sequence()

ipro = 0
rf_phase = 0
rf_inc = 0
#for i in range(nProj + nDummy): //Real one
rf.phase_offset = rf_phase / 180 * np.pi
rf.phase_offset = rf_phase / 180 * np.pi
adc.phase_offset = rf_phase / 180 * np.pi
rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
rf_phase = np.mod(rf_phase + rf_inc, 360.0)

seq.add_block(rf)
seq.add_block(adc,pp.scale_grad(grad = g_arb1, scale=1),pp.scale_grad(grad = g_arb1_y, scale=0.5),pp.scale_grad(grad = g_arb1_z, scale=0.75))
seq.add_block(adc,pp.scale_grad(grad = g_arb1, scale=-1),pp.scale_grad(grad = g_arb1_y, scale=-0.5),pp.scale_grad(grad = g_arb1_z, scale=-0.75))
seq.add_block(pp.make_delay(np.ceil((TR - (pp.calc_duration(rf) + 2*pp.calc_duration(g_arb1)))/ seq.grad_raster_time)*seq.grad_raster_time))

seq.add_block(rf)
seq.add_block(pp.scale_grad(grad = g_arb1, scale=1),pp.scale_grad(grad = g_arb1_y, scale=0.5),pp.scale_grad(grad = g_arb1_z, scale=0.75))
#seq.add_block(pp.scale_grad(grad = g_arb1, scale=-1),pp.scale_grad(grad = g_arb1_y, scale=-0.5),pp.scale_grad(grad = g_arb1_z, scale=-0.75))
seq.add_block(pp.make_delay(np.ceil((TR - (pp.calc_duration(rf) + 2*pp.calc_duration(g_arb1)))/ seq.grad_raster_time)*seq.grad_raster_time))

seq.add_block(rf)
seq.add_block(adc,pp.scale_grad(grad = g_arb0, scale=1),pp.scale_grad(grad = g_arb0_y, scale=0.5),pp.scale_grad(grad = g_arb0_z, scale=0.75))
seq.add_block(adc,pp.scale_grad(grad = g_arb0, scale=-1),pp.scale_grad(grad = g_arb0_y, scale=-0.5),pp.scale_grad(grad = g_arb0_z, scale=-0.75))
#seq.add_block(pp.make_delay(np.ceil((TR - (pp.calc_duration(rf) + 2*pp.calc_duration(g_arb0)))/ seq.grad_raster_time)*seq.grad_raster_time))


#extended plus
seq.add_block(rf)
seq.add_block(adc,pp.scale_grad(grad = g_tot, scale=1),pp.scale_grad(grad = g_tot_y, scale=0.5),pp.scale_grad(grad = g_tot_z, scale=0.75))
seq.add_block(adc,pp.scale_grad(grad = g_tot, scale=-1),pp.scale_grad(grad = g_tot_y, scale=-0.5),pp.scale_grad(grad = g_tot_z, scale=-0.75))
seq.add_block(pp.make_delay(np.ceil((TR - (pp.calc_duration(rf) + 2*pp.calc_duration(g_tot)))/ seq.grad_raster_time)*seq.grad_raster_time))


# Trapezoids plus

seq.add_block(rf_trap, gz_trap)
gy_trap_pre = pp.make_trapezoid(
    channel='y',
    area=phase_areas_trap[0],
    duration=pp.calc_duration(gx_trap_pre),
    system=system,
)

seq.add_block(gx_trap_pre, gy_trap_pre, gz_trap_reph)
#seq.add_block(pp.make_delay(delay_TE))
seq.add_block(gx_trap, adc_trap)
gy_trap_pre.amplitude = -gy_trap_pre.amplitude
seq.add_block(pp.make_delay(delay_TR), gx_trap_spoil, gy_trap_pre, gz_trap_spoil)


# Trapezoids minus

gx_trap.amplitude *= -1
gx_trap_pre.amplitude *= -1
gz_trap.amplitude *= -1
gz_trap_reph.amplitude *= -1
gx_trap_spoil.amplitude *= -1
gz_trap_spoil.amplitude *= -1

gx_trap.area *= -1
gx_trap_pre.area *= -1
gz_trap.area *= -1
gz_trap_reph.area *= -1
gx_trap_spoil.area *= -1
gz_trap_spoil.area *= -1

seq.add_block(rf_trap, gz_trap)
gy_trap_pre = pp.make_trapezoid(
    channel='y',
    area=phase_areas_trap[0],
    duration=pp.calc_duration(gx_trap_pre),
    system=system,
)

gy_trap_pre.amplitude *= -1
gy_trap_pre.area *= -1

seq.add_block(gx_trap_pre, gy_trap_pre, gz_trap_reph)
#seq.add_block(pp.make_delay(delay_TE))
seq.add_block(gx_trap, adc_trap)
gy_trap_pre.amplitude = -gy_trap_pre.amplitude
seq.add_block(pp.make_delay(delay_TR), gx_trap_spoil, gy_trap_pre, gz_trap_spoil)


# %%
#seq.plot(grad_disp="mT/m",show_blocks=True,time_range=(TR*(nDummy+1),TR*(nDummy+3)),time_disp="s",label="SET,PAR")

# %%
if FLAG_PLOT:
  seq.plot(grad_disp="mT/m",show_blocks=True,time_disp="s",label="SET,PAR")


# %%
seq.check_timing()



# %%
seq.set_definition("Name","UTE_"+version_sequence)
seq.set_definition("version_sequence",version_sequence)
# Paramètres système
seq.set_definition("max_grad",system.max_grad)
seq.set_definition("max_slew",system.max_slew)
seq.set_definition("rf_ringdown_time",system.rf_ringdown_time)
seq.set_definition("rf_dead_time",system.rf_dead_time)
seq.set_definition("adc_dead_time",system.adc_dead_time)

# paramètres modifiable
seq.set_definition("FOV",fov)

N2 = [N,N]
seq.set_definition("matrix",N2)
seq.set_definition("alpha",alpha)
seq.set_definition("TR",TR*1e3)
seq.set_definition("nProj",nProj)
seq.set_definition("reord_traj",reord_traj)
seq.set_definition("nSegment",nSeg)
seq.set_definition("SGPoints",SGPoints)

## param supp
seq.set_definition("rise_time",riseT)
seq.set_definition("dwell",dwell)
seq.set_definition("nDummy",nDummy)
seq.set_definition("rf_len",rf_len)
seq.set_definition("rf_spoiling_inc",rf_spoiling_inc)
seq.set_definition("ro_spoil",ro_spoil)
seq.set_definition("ro_os",ro_os)

# param faciliter la reco
seq.set_definition("nSample",Ns)
seq.set_definition("2D", "2D")

# %%
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%Hh%Mm")
print("date and time =", dt_string)


# %%
if FLAG_WRITE_SEQ:
  output_path = '/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/python/Test_unitaire/abstract/ouput'
  seq_filename = "Arb-1_plus.seq"
  seq.write(str(output_path+"/"+seq_filename),remove_duplicates=False)



