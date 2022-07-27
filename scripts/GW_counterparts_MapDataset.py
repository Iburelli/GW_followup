import numpy as np
import matplotlib.pyplot as plt
from regions import CircleSkyRegion
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astropy.wcs import WCS
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.datasets import Datasets,  MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import (TemplateSpectralModel,
                                     SkyModel,
                                     PointSpatialModel,
                                     FoVBackgroundModel,
                                     Models,
                                     GaussianSpatialModel,
                                     PowerLawSpectralModel)
from gammapy.estimators import TSMapEstimator
import pandas as pd
from gammapy.estimators.utils import find_peaks
from gammapy.irf import load_cta_irfs
from gammapy.makers import ReflectedRegionsBackgroundMaker, MapDatasetMaker, SafeMaskMaker
import warnings
import argparse
import yaml
import os
with warnings.catch_warnings():
    from gammapy.data import Observation
from gammapy.maps import MapAxis, RegionNDMap, Map

parser = argparse.ArgumentParser(description='')
parser.add_argument('-f', '--config', required=True, type=str, help='configuration yaml file')
# configuration file
cf = parser.parse_args().config
# load params configuration from cf
with open(cf) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# -----------------------------------------------------------------
def irf_selection(site,z,delta_t):
    if delta_t < 94.9 * 60:   #times are converted in SECONDS
        irf_duration = '0.5h'
    elif 94.9 * 60 < delta_t < 15.8 * 60 * 60:
        irf_duration = '5h'
    elif delta_t > 15.8 * 60 * 60:
        irf_duration = '50h'

    if z == 60:
        energy = 0.110          #energies are in TeV
        #sim_e_max = 5.6234

    elif z == 40:
        energy = 0.04
        #sim_e_max = 5.6234
    else:
        energy = 0.03
        #sim_e_max = 10.


    name = (f'{site}_z{z}_{irf_duration}')
    return [name , energy]
# -------------------------------------------------------------

inputdir=cfg['input']['dir']
eventid=cfg['input']['evid']
pointings_list=cfg['input']['pointings']
vis_table = cfg['input']['vis_tab']
caldb=cfg['input']['caldb']

filename = f'{inputdir}/{eventid}.fits'

hdul = fits.open(filename)
hdr=hdul[0].header
df = pd.read_csv (pointings_list,
                        header=0,
                        sep=',',
                        engine='python')


# reading trigger time
if 'GRBJD' not in hdr.keys():
    print ('\nReading trigger time from pointings file')

    time = (Time(df['Observation_Time_UTC'][0], scale='utc' ,format='isot') - df['Delay[s]'][0]*u.s)
    print(f'\tTrigger time for event {eventid} : {time}')
    trigger=time.jd
    print(f'\tTrigger time jd units: {trigger}')

else:

    trigger=hdr['GRBJD']

# defining on region and pointing direction
grb_radec = SkyCoord(ra=hdr['RA'] * u.deg, dec=hdr['DEC'] * u.deg, frame='icrs')   # source coordinates
spatial_model=PointSpatialModel(lon_0=grb_radec.ra, lat_0=grb_radec.dec, frame="icrs")
# reading Energy from fits (to be transformed in a property of a GRB object.
grb_Eval = []
grb_Eval = Table.read(hdul,hdu=1)["Energies"].quantity
grb_Eval     = grb_Eval.to(u.TeV)

# Reading time intervals from fits file (to be transformed in a property of a GRB object)
grb_tval=[]
grb_tval = Table.read(hdul,hdu=2)["Final Time"].quantity
lvtm=[]
lvtm.append(grb_tval[0])
for i in range(1,len(grb_tval)):
    lvtm.append(grb_tval[i]-grb_tval[i-1])


# reading flux form fits files:
# unabsorbed flux, by now will not be considered as a problem: sources are
# closeby and than only mildly absorbed
flux=[]
flux = Table.read(hdul,hdu=3)
icol_E  = len(flux.colnames)             # column number - energy
jrow_t  = len(flux[flux.colnames[0]])    # row number - time
magnify= 1e3                    # needed to convert flux from 1/GeV to 1/TeV
flux_unit = u.Unit("1 /(cm2 TeV s)")
# Grb fluxval must have the dimension of time x energy,
# the same as flux in this case
# ----------------------  TRUE
grb_fluxval = np.zeros( (jrow_t,icol_E) )*flux_unit
#print(grb_fluxval.shape)
for i in range(0,icol_E):
    for j in range(0,jrow_t):
        f = flux[j][i]
        grb_fluxval[j][i] = magnify*f*flux_unit
# ----------------------
# creating a spectral model for each one of the time intervals
non_absorbed=[]

for i,t in enumerate(grb_tval):

    non_absorbed.append(TemplateSpectralModel(energy = grb_Eval.astype(float),
                                        values = grb_fluxval[i],
                                        interp_kwargs={"values_scale": "log"}))


data = np.load(vis_table, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
events = list(data.keys())
sites = list(data[events[0]].keys())

for ii in range(len(df['Observation_Time_UTC'])):
    observation_onset = Time(df['Observation_Time_UTC'][ii], scale='utc' ,format='isot')
    observation_onset=observation_onset.jd
    pointing = SkyCoord(df['RA[deg]'][ii],df['DEC[deg]'][ii] , unit="deg", frame="icrs")
    duration = df['Duration[s]'][ii]
    start=[]
    stop=[]
    z=[]
    models=[]

    for event in events:
        if event == eventid:
            for site in sites:
                if site ==df['Observatory'][0]:

                    for night in data[event][site].keys():
                        if night == 'night01':  # I could also simulate all 10 nights
                                                # but would not make sense, and would take really ong
                            # start of visibility interval


                            if observation_onset == trigger :
                                t_obs_start = data[event][site][night]['irfs']['start'][0]

                            else:
                                t_obs_start = observation_onset

                            t_obs_start = (t_obs_start - trigger)*86400
                            #print('t_obs_start:', t_obs_start)

                            # stop of visibility interval

                            t_night_stop = t_obs_start + df['Duration[s]'][ii]
                            #print('t_night_stop:', t_night_stop)

                            #print (t_night_stop - t_obs_start)

                            #  ---------------
                            t_obs_stop = t_night_stop




                            for j in range(len(lvtm)-1):
                                #if j==0:

                                for i in range(len(data[event][site][night]['irfs']['zref'])):
                                    #if i==0:
                                        #print(i)
                                        if data[event][site][night]['irfs']['zref'][i]==-9.0:
                                            break

                                        #-----------------SIMULATION TIMES


                                        t_min = data[event][site][night]['irfs']["start"][i]
                                        t_max = data[event][site][night]['irfs']["stop"][i]
                                        zenith=data[event][site][night]['irfs']["zref"][i]

                                        # converting times from jd to seconds from trigger

                                        t_min = (t_min - trigger) * 86400
                                        #print(t_min)
                                        t_max = (t_max - trigger) * 86400
                                        #print(t_max)
                                        #print(zenith_angle, t_min, t_max)
                                        if grb_tval[j+1].value < t_obs_start:
                                            continue

                                        t_slice_start = grb_tval[j].value

                                        if t_slice_start < t_obs_start:
                                            t_slice_start = t_obs_start

                                        #print('t_slice_start:', t_slice_start)
                                        #print(t_obs_stop , t_slice_start)

                                        if t_obs_stop - t_slice_start <= 0:
                                            break

                                        t_slice_stop  = grb_tval[j+1].value
                                        #print('t_slice_stop:', t_slice_stop)
                                        #print('t_max:',t_max)

                                        if t_slice_stop <= t_min:
                                            #print ('ok')
                                            continue

                                        if t_slice_start >= t_max:
                                            continue
                                        #print(t_slice_start,t_min)

                                        t_slice_start= max(t_slice_start, t_min)
                                        t_slice_stop = min (t_slice_stop, t_max)
                                        #print(j,t_slice_start,t_slice_stop, t_slice_stop-t_slice_start)


                                        if t_slice_stop >= t_obs_stop:
                                            t_slice_stop = t_obs_stop


                                        start.append(t_slice_start)
                                        stop.append(t_slice_stop)
                                        z.append(zenith)
                    if len(start)==0:
                        print('entering the loop')

                        start.append(df['Delay[s]'][kk])
                        stop.append(df['Delay[s]'][kk] + df['Duration[s]'][kk])
                        m=np.abs((df['ZenIni[deg]'][kk]-df['ZenEnd[deg]'][kk])/2)
                        if m<33:
                            zd=20
                        elif m>33 and m< 54:
                            zd=40
                        elif m >54 and m< 80:
                            zd=60
                        z.append(zd)

                    for i in range(len(start)):
                        for j in range(len(grb_tval)-1):
                            if stop[i] <= grb_tval[j+1].value and start[i]>=grb_tval[j].value:
                                model = SkyModel(spectral_model=non_absorbed[i],spatial_model=spatial_model, name=f"{eventid}")
                                models.append (model)
                    name_irf=[]
                    sim_e_min=[]
                    for event in events:
                        if event == eventid:
                            for site in sites:
                                if site == df['Observatory'][0]:
                                    first_night_start = data[event][site]['night01']['irfs']['start'][0]
                                    first_night_start  = (first_night_start - trigger)*86400
                                    for i in range(len(start)):
                                        t_start=start[i]
                                        t_stop = stop[i]
                                        zenith_angle = z[i]
                                        # ----------------------------------------time selection for IRF
                                        delta_t_irf = t_stop - first_night_start
                                        #print(delta_t_irf)
                                        name_irf.append(irf_selection(site, zenith_angle, delta_t_irf)[0])
                                        sim_e_min.append(irf_selection(site, zenith_angle, delta_t_irf)[1])
                    irfs=[]
                    for i in range(len(name_irf)):
                        path = f"{caldb}/{name_irf[i]}"

                        # iterate through all file
                        if '$' in path:
                            path= os.path.expandvars(path)
                        for file in os.listdir(path):
                            # Check whether file is in text format or not
                            if file.endswith(".gz"):
                                file_path = f"{path}/{file}"
                        irf = load_cta_irfs(file_path)
                        irfs.append(irf)
                    n_obs = len(start)
                    # from SoHappy
                    maker=[]
                    empty=[]
                    erec_edges = np.asarray([3.00000000e+01, # LST 20°
                                            #3.16227766e+01, # too close fro previous edge
                                            4.00000000e+01, # LST 40°
                                            5.62341325e+01,
                                            #1.00000000e+02, # too close from next edge
                                            1.10000000e+02, # LST 60, MST 20° and 40°
                                            # 1.77827941e+02,  # too close from next edge
                                            2.0e+02, # too close from next edge
                                            #2.5e+02, # MST 60°
                                            3.16227766e+02,
                                            5.62341325e+02,
                                            1.00000000e+03,
                                            1.77827941e+03,
                                            3.16227766e+03,
                                            5.62341325e+03,
                                            1.00000000e+04])*u.GeV
                    energy_axis      = MapAxis.from_edges(erec_edges.to("TeV").value,
                                       unit="TeV",
                                       name="energy",
                                       interp="log")


                    energy_axis_true  = MapAxis.from_energy_bounds(0.01*u.TeV,
                                                                     50.0*u.TeV,
                                                                     nbin = 20,
                                                                     per_decade=True,
                                                                     name="energy_true")


                    geom              = WcsGeom.create(skydir=pointing,
                                                        binsz=0.02,
                                                        width=(6, 6),
                                                        frame="icrs",
                                                        axes=[energy_axis])
                    datasets = Datasets()
                    # -------------------
                    stacked = MapDataset.create(geom=geom,energy_axis_true=energy_axis_true, name=eventid)
                    # --------------
                    #factors=[]
                    #ns=[]
                    #nb=[]
                    # --------------
                    for idx in range(n_obs):
                        #print(idx)
                        irf   = irfs[idx]

                        bkg_model = FoVBackgroundModel(dataset_name=f"{eventid}")

                        model = Models([models[idx], bkg_model])


                        # -------------- cannot be moved outside cycle, it's  IRF dependent
                        eirf_min    = min(irf["aeff"].axes["energy_true"].edges)

                        empty = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, name=f"dataset-{idx}")

                        maker =  MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])

                        #-----------------------
                        with warnings.catch_warnings(): # because of t_trig
                            warnings.filterwarnings("ignore")
                            obs = Observation.create(
                                pointing=pointing,
                                livetime=(stop[idx]-start[idx])*u.s,
                                reference_time = Time(trigger * u.day, format='jd')+start[idx]*u.s,
                                deadtime_fraction = 0 ,
                                irfs=irf,
                                obs_id=idx)

                        # -----------------
                        dataset = maker.run(empty, obs)
                        # -----------------
                        dataset.models = model
                        #-----------------------
                        dataset.mask_fit = dataset.counts.geom.energy_mask(energy_min=eirf_min , energy_max=None)

                        dataset.fake(random_state=42)

                        datasets.append(dataset)
                        #ns.append(dataset.counts.data[dataset.mask_safe].sum())
                        #nb.append(dataset.background.data[dataset.mask_safe].sum())
                        stacked.stack(dataset)

                    stacked.write(f'pointing_{ii}_{eventid}.fits',overwrite=True)
                    spatial_model =PointSpatialModel()
                    spectral_model = PowerLawSpectralModel()
                    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model) # dummy model to look for sources

                    ts_image_estimator = TSMapEstimator(model,
                                                kernel_width="0.5 deg",
                                                selection_optional=[],
                                                downsampling_factor=2,
                                                sum_over_energy_groups=False,
                                                energy_edges=[0.11, 10] * u.TeV,
                                            )
                    images_ts = ts_image_estimator.run(stacked)
                    sources = find_peaks(
                                            images_ts["sqrt_ts"],
                                            threshold=5,
                                            min_distance="0.2 deg",
                                        )
                    if len(sources)>0:
                        source_pos = SkyCoord(sources["ra"], sources["dec"])
                        print(source_pos)
                        images_ts["sqrt_ts"].plot(add_cbar=True)

                        plt.gca().scatter(
                            source_pos.ra.deg,
                            source_pos.dec.deg,
                            transform=plt.gca().get_transform("icrs"),
                            color="none",
                            edgecolor="white",
                            marker="o",
                            s=200,
                            lw=1.5,
                        );
                        print(f'\n Pointing number {ii +1}')
                        print('\tI detected the following sources')
                        print(sources)
                        print(f'\tAngular distance of src from camera center: {pointing.separation(grb_radec)}')
                        print(f"\tPointing delay: {df['Delay[s]'][ii]}")
                        print(f"Exposure time: {df['Duration[s]'][ii]}")
                    else:
                        print(f'\nPointing number {ii+1}')
                        print('\tNo source detected')
                        print(f'\tAngular distance of src from camera center: {pointing.separation(grb_radec)}')
                        print(f"\tPointing delay: {df['Delay[s]'][ii]}")
                        print(f"Exposure time: {df['Duration[s]'][ii]}")
