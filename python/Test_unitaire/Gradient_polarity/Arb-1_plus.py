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
FLAG_WRITE_SEQ = True
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

slewRate_up1 = gro.amplitude/gro.rise_time
step1 = slewRate_up1*(system.grad_raster_time/2)
wave1 = np.arange(0,gro.amplitude+step1, step1)
flat1 = gro.flat_time/(system.grad_raster_time/2)

wave2 = np.ones(int(flat1))*gro.amplitude
wave12 = np.concatenate((wave1,wave2))

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
wave15 = np.concatenate((wave14, wave5))

print("wave15", len(wave15))
g_spoil_opt.delay=gro_sum.shape_dur

# %%
# combine gradients
g_tot = pp.add_gradients([gro_sum,g_spoil_opt],system=system)
g_tot

g_arb = pp.make_arbitrary_grad('x', wave15, system=system,first=0,last=0, oversampling=True)

# %%
# add SG delay
g_tot.delay = SGPoints*dwell
g_arb.delay = SGPoints*dwell
# %%
# Delay_TR
delay_TR = TR - (pp.calc_duration(rf)+
                 pp.calc_duration(g_tot))




# %%
scale_x = np.zeros((nProj),dtype=float)
scale_y = np.zeros((nProj),dtype=float)
scale_z = np.zeros((nProj),dtype=float)



phi = np.linspace(0,2*pi,nProj)

for ipro in range(nProj): # 2D
    scale_x[ipro] = np.cos(phi[ipro])
    scale_y[ipro] = np.sin(phi[ipro])
    scale_z[ipro] = 0

scale_x

# %% [markdown]
# 2. reorder with 1D golden angle

# %%
if reord_traj == "golden":
    print("reorder : gold")
    scale_x_tmp = scale_x.copy()
    scale_y_tmp = scale_y.copy()
    scale_z_tmp = scale_z.copy()

    index = 0
    for ipro in range(nProj): # 2D
        if ipro == (nProj-1):
            index = 0
        else:
            index = int(np.ceil(np.mod((nProj-(ipro+1))*((ipro+1)*(np.sqrt(5.0)-1) / 3.0),nProj-(ipro+1))))

        scale_x[ipro] = scale_x_tmp[index]
        scale_y[ipro] = scale_y_tmp[index]
        scale_z[ipro] = scale_z_tmp[index]

        np.delete(scale_x_tmp,index)
        np.delete(scale_y_tmp,index)
        np.delete(scale_z_tmp,index)
elif reord_traj == "segment":
    print("reorder : segment")
    scale_x_tmp = scale_x.copy()
    scale_y_tmp = scale_y.copy()
    scale_z_tmp = scale_z.copy()

    index = 0
    for seg in range(int(nProj/nSeg)): # 2D
        #print(seg)
        for ipro in np.arange(seg,nProj,int(nProj/nSeg)): # 2D
            #print(ipro)
            scale_x[index] = scale_x_tmp[ipro]
            scale_y[index] = scale_y_tmp[ipro]
            scale_z[index] = scale_z_tmp[ipro]

            index = index + 1



# %% [markdown]
# # Write real sequence

# %%
g_arb_y = copy.deepcopy(g_arb)
g_arb_z = copy.deepcopy(g_arb)
g_arb_y.channel='y'
g_arb_z.channel='z'

# %%
seq=pp.Sequence()

ipro = 0
rf_phase = 0
rf_inc = 0
#for i in range(nProj + nDummy): //Real one
for i in range(nProj):
  rf.phase_offset = rf_phase / 180 * np.pi
  rf.phase_offset = rf_phase / 180 * np.pi
  adc.phase_offset = rf_phase / 180 * np.pi
  rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
  rf_phase = np.mod(rf_phase + rf_inc, 360.0)

  seq.add_block(rf)
  seq.add_block(adc,pp.scale_grad(grad = g_arb, scale=scale_x[ipro]),pp.scale_grad(grad = g_arb_y, scale=scale_y[ipro]),pp.scale_grad(grad = g_arb_z, scale=scale_y[ipro]))
  ipro = ipro + 1

  seq.add_block(pp.make_delay(delay_TR))

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
  output_path = '/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/python/Test_unitaire/Gradient_polarity/ouput'
  seq_filename = "Arb-1_plus.seq"
  seq.write(str(output_path+"/"+seq_filename),remove_duplicates=False)



