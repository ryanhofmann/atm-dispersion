#!/usr/bin/env python3

# Load all necessary modules
import pickle
from astropy.convolution import convolve, CustomKernel, Box1DKernel
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import interpolate
import sunpy as sp
from sunpy import coordinates as coord
from astropy.time import Time
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.coordinates import EarthLocation, AltAz, TETE
from astropy import coordinates
import astropy.units as u
from datetime import datetime
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('./python/')

print("All modules loaded")


# Determine atmospheric density

def atmospheric_density(
        temperature=20*u.deg_C,
        pressure_pa=100000*u.Pa,
        humidity=75,
        xc=380,
        force_xw=0,
        water_vapor=False,
        dry_air=False,
        verbose=0):
    """Return the atmospheric density

    Typical parameters for atmospheric values set as defaults.
    """

    TT_C = temperature.value
    TT_K = TT_C + 273.15
    pressure_pa = pressure_pa.to(u.Pa).value  # keep units in pascals
    humidity_partial = humidity/100.
    if verbose >= 2:
        print("Temp: ", TT_C, "°C \nPressure: ", pressure_pa,
              " Pa \nHumidity: ", humidity, "% \nCO2: ", xc, " ppm")

    # *************** Constants ****************
    # from Ciddor 1996, Appendix A:
    AA = 1.2378847*10**-5  # K^(-2)
    BB = -1.9121316*10**-2  # K^(-2)
    CC = 33.93711047         #
    DD = -6.3431645*10**3  # K

    alpha = 1.00062
    beta = 3.14 * 10**-8  # Pa^(-1)
    gamma = 5.6 * 10**-7  # °C^(-2)

    a0 = 1.58123*10**-6  # K Pa^(-1)
    a1 = -2.9331*10**-8  # Pa^(-1)
    a2 = 1.1043*10**-10  # K^(-1) Pa^(-1)
    b0 = 5.707*10**-6  # K Pa^(-1)
    b1 = -2.051*10**-8  # Pa^(-1)
    c0 = 1.9898*10**-4  # K Pa^(-1)
    c1 = -2.376*10**-6  # Pa^(-1)
    d = 1.83*10**-11  # K^2 Pa^(-2)
    e = -0.765*10**-8  # K^2 Pa^(-2)

    # from Ciddor 1995, Section 3
    # gas constant:
    R = 8.314510  # J mol^(-1) K^(-1)
    # molar mass of water vapor:
    Mw = 0.018015  # kg/mol
    # molar mass of dry air containing a CO2 concentration of xc ppm:
    Malpha = (10**-3)*(28.9635 + (12.011*10**-6)*(xc-400))

    # ***************End Constants*****************

    # saturation vapor pressure of water vapor in air at temperature T,
    # from Ciddor 1996 Section 3:
    svp = np.exp(AA*TT_K**2 + BB*TT_K + CC + DD/TT_K)

    # enhancement factor of water vapor in air, whatever that is:
    f = alpha + beta*pressure_pa + gamma*TT_C**2

    if force_xw == 0:
        xw = f*humidity_partial*svp/pressure_pa
    else:
        xw = force_xw  # molar fraction of water vapor

    # from Ciddor 1996 Appendix:
    Z = (1 - (pressure_pa/TT_K)*(a0 + a1*TT_C + a2*TT_C**2
                                 + (b0 + b1*TT_C)*xw
                                 + (c0 + c1*TT_C)*xw**2)
         + ((pressure_pa/TT_K)**2)*(d + e*xw**2))

    if water_vapor:
        density = (pressure_pa*Mw*xw/(Z*R*TT_K))
    elif dry_air:
        density = (pressure_pa * Malpha/(Z*R*TT_K))*(1 - xw)
    else:
        density = (pressure_pa * Malpha/(Z*R*TT_K))*(1 - xw*(1 - Mw/Malpha))

    if verbose >= 2:
        print("svp: ", svp, '\nf: ', f, '\nxw: ', xw,
              '\nZ: ', Z, '\ndensity: ', density)

    atmosphere_values = {
        'R': R,
        'Z': Z,
        'Ma': Malpha,
        'Mw': Mw,
        'svp': svp,
        'f': f,
        'density': density,
        'TT_C': TT_C,
        'TT_K': TT_K,
        'pressure_pa': pressure_pa,
        'humidity': humidity,
        'xw': xw}

    return(density)


# Use density to get refractivity

def refractivity(
        wavelength_nm=np.array([633.0])*u.nm,
        temperature=20*u.deg_C,
        pressure_pa=100000*u.Pa,
        humidity=75,
        xc=380,
        verbose=0):
    """Return the refractivity at a given wavelength

    Typical parameters for atmospheric values set as defaults.
    Note that refractivity  = index of refraction - 1
    """

    wavelength_mic = wavelength_nm.to(u.micron).value

    # convert wavelengths in air to vacuum wavelengths
    # [lambda(air) = lambda(vacuum)/n(air)]
    # using mean index of refraction of air = 1.00027
    wavelength_vac = wavelength_mic*1.00027
    wavenumber = 1/wavelength_vac

    # *****************  Constants  ******************
    # from Ciddor 1996, Appendix A
    # originally from Peck and Reeder 1962:
    k0 = 238.0185  # microns^(-2)
    k1 = 5792105.  # microns^(-2)
    k2 = 57.362  # microns^(-2)
    k3 = 167917.  # microns^(-2)

    # originally from Owens 1967:
    w0 = 295.235  # microns^(-2)
    w1 = 2.6422  # microns^(-2)
    w2 = -0.032380  # microns^(-4)
    w3 = 0.004028  # microns^(-6)
    # *************  End Constants  ******************

    # refractivity of air at 15°C, 101325 Pa, 0% humidity, and a fixed
    # 450 ppm of CO2 from Ciddor 1996 Eq. 1:
    nas = (10**-8)*(k1/(k0 - wavenumber**2) + k3/(k2 - wavenumber**2)) + 1

    # refractivity of air at 15°C, 101325 Pa, 0% humidity, and a variable
    # xc pmm of CO2 from Ciddor 1996 Eq. 2:
    naxs = (nas - 1)*(1 + (0.534*10**-6)*(xc - 450)) + 1

    # refractivity of water vapor at 20°C, 1333 Pa, 0% humidity
    # correction actor derived by Ciddor 1996 by fitting to measurements:
    cf = 1.022
    # from Ciddor 1996 Eq. 3:
    nws = (10**-8)*cf*(w0 + w1*wavenumber**2
                       + w2*wavenumber**4 + w3*wavenumber**6) + 1

    # density of dry air at standard conditions:
    density_axs = atmospheric_density(
        15*u.deg_C, 101325*u.Pa, 0, xc, dry_air=True)
    # density of water vapor at standard conditions:
    density_ws = atmospheric_density(
        20*u.deg_C, 1333*u.Pa, 100, xc, force_xw=1)

    # density of dry air at input conditions:
    density_a = atmospheric_density(
        temperature, pressure_pa, humidity, xc, dry_air=True)
    # density of water vapor at input conditions:
    density_w = atmospheric_density(
        temperature, pressure_pa, humidity, xc, water_vapor=True)
    if verbose >= 1:
        print("density a - ", density_a, density_axs, density_a/density_axs)
        print("density w - ", density_w, density_ws, density_w/density_ws)

    # from Ciddor 1996 Eq. 5:
    nprop_a = (density_a/density_axs)*(naxs - 1)
    nprop_w = (density_w/density_ws)*(nws - 1)
    nprop = nprop_a + nprop_w

    if verbose >= 1:
        print("n(axs): ", (naxs - 1)*10**8,
              "\nn(ws): ", (nws - 1)*10**8,
              "\nrho(a/axs): ", (density_a/density_axs),
              "\nrho(w/ws): ", (density_w/density_ws),
              "\nn(prop): ", nprop*10**8)
    if verbose >= 2:
        print("n(air): ", (density_a/density_axs)*(naxs - 1)*10**8,
              "\nn(water): ", (density_w/density_ws)*(nws - 1)*10**8)

    return(nprop)


# Main atmospheric refractivity function

def atmospheric_refraction(
        wavelength=np.array([400, 500, 600, 700, 800])*u.nm,
        input_times=np.array([1]),
        input_alt=None,
        input_az=None,
        input_ha=None,
        observer_location=EarthLocation(lon=-156.25, lat=20.71, height=3070),
        air_temp=20*u.deg_C,
        air_pressure=100000.*u.Pa,
        humidity=75.,
        co2_conc=415.,
        verbose=0):
    """Return the wavelengths-dependent atmospheric refraction

    Typical parameters for atmospheric values set as defaults.
    Parallactic angle in degrees and refraction magnitude in arcseconds
    are also returned.

    """

    arcsec_conversion = 206265.
    num_waves = wavelength.size
    wavelength.astype(float)

    # setting default time to now in Julian dates
    if np.alltrue(input_times == np.array([1])):
        input_times = Time(np.array([datetime.utcnow()]), scale='utc')
        input_times.format = 'jd'
    else:
        input_times = Time(input_times, format='jd', scale='utc')

    num_times = input_times.size

    lat = observer_location.geodetic.lat.value

    refrac = refractivity(wavelength, air_temp, air_pressure,
                          humidity, co2_conc, verbose=verbose)
    if num_waves == 1:
        refrac = np.array([refrac])

    if input_alt is not None and input_ha is not None:
        alt_all = input_alt
        az_all = input_az
        ha_all = input_ha
    elif input_alt is not None and input_az is not None:
        alt_all = input_alt
        az_all = input_az
        dec = np.rad2deg(np.arcsin(np.sin(alt_all*u.deg) * np.sin(lat*u.deg)
                                   + np.cos(alt_all*u.deg) * np.cos(lat*u.deg)
                                     * np.cos(az_all*u.deg)).value)
        ha_all = np.rad2deg(-np.arcsin(np.sin(az_all*u.deg)
                                       * np.cos(alt_all*u.deg)
                                       / np.cos(dec*u.deg)).value)
        ha_all = (ha_all + 180) % 360 - 180
    else:
        local_sidereal = input_times.sidereal_time(
            'apparent', observer_location.geodetic.lon)

        # get the Sun's RA and Dec, then print them
        with erfa_astrom.set(ErfaAstromInterpolator(1440 * u.min)):
            sunpos = coordinates.get_sun(input_times)
            frame_TETE = TETE(obstime=input_times, location=observer_location)
            sunpos_TETE = sunpos.transform_to(frame_TETE)
        if verbose == 1:
            print(sunpos.ra, sunpos.dec, sunpos.obstime)

        # Get hour angle, altitude and azimuth
        # The rest of the program only uses the hour angle and altitude
        ha_all = ((local_sidereal - sunpos_TETE.ra).deg + 180) % 360 - 180
        with erfa_astrom.set(ErfaAstromInterpolator(1440 * u.min)):
            frame_obstime = AltAz(obstime=input_times, location=observer_location)
            sunpos_altaz = sunpos.transform_to(frame_obstime)
        alt_all = sunpos_altaz.alt.deg
        az_all = sunpos_altaz.az.deg

    # continue with calculations
    beta = 0.001254*(273.15 + air_temp.value)/273.15
    coeff_a = refrac*(1 - beta)
    coeff_b = refrac*(beta - refrac/2.)

    # calculate the magnitude of the refraction for each time and wavelength
    refraction_calc = np.ones((num_times, num_waves))
    for wv in range(num_waves):
        refraction_wv = (coeff_a[wv]*np.tan(np.radians(90 - alt_all)))
        - (coeff_b[wv]*(np.tan(np.radians(90 - alt_all)))**3)

        refraction_wv = refraction_wv*arcsec_conversion
        refraction_calc[:, wv] = refraction_wv
    # find the parallactic angle

    # get everything in degrees

    parallactic_angle_sin = np.clip(np.sin(ha_all*u.deg)
                                    / np.sin((90 - alt_all)*u.deg)
                                    * np.sin((90 - lat)*u.deg), -1, 1)
    parallactic_angle = np.rad2deg(np.arcsin(parallactic_angle_sin)).value

    if verbose == 1:
        print("\nInput Time(s) in Julian dates: ", input_times)
        print("\nSun's RA: ", sunpos.ra.degree)
        print("Sun's Dec: ", sunpos.dec.degree)
        print("Local Sidereal Time: ", local_sidereal)
        print('\nHour Angle: ', ha_all)
        print("Altitude: ", alt_all)
        print("Azimuth: ", sunpos_altaz.az.deg)
        print()
        for time, refractions in zip(input_times, refraction_calc):
            print("Refraction for Julian Date ", time, ": ", refractions)
        print()
        for time, angles in zip(input_times, parallactic_angle):
            print("Parallactic Angle for Julian Date ", time, ": ", angles)

    atmospheric_refraction = {
        'refraction_mag (arcsec)': refraction_calc[:, :],
        'parallactic_angle (degrees)': parallactic_angle[:]}
    coords = {
        'altitude (degrees)': alt_all,
        'azimuth (degrees)': az_all,
        'hour angle (degrees)': ha_all}
    return(atmospheric_refraction, input_times, coords)


# Calculate refraction offsets in Solar NS-EW coordinate system

def offsets(
        wavelength=np.array([400, 500, 600, 700, 800])*u.nm,
        input_times=np.array([1]),
        input_alt=None,
        input_az=None,
        input_ha=None,
        observer_location=EarthLocation(lon=-156.25, lat=20.71, height=3070),
        air_temp=20.*u.deg_C,
        air_pressure=100000.*u.Pa,
        humidity=75.,
        co2_conc=380.,
        verbose=0):
    """Computes Heliocentric shifts due to refraction

    Typical parameters for atmospheric values set as defaults.
    Computes North-South and East-West offsets in Heliocentric coordinates

    """

    # setting default time to now in Julian dates
    if np.alltrue(input_times == np.array([1])):
        input_times = Time(np.array([datetime.utcnow()]), scale='utc')
        input_times.format = 'jd'
    else:
        input_times = Time(input_times, format='jd', scale='utc')
    num_times = input_times.size

    (refraction_atm, input_times, coords) = atmospheric_refraction(
        wavelength=wavelength,
        input_times=input_times,
        input_alt=input_alt,
        input_az=input_az,
        input_ha=input_ha,
        observer_location=observer_location,
        air_temp=air_temp,
        air_pressure=air_pressure,
        humidity=humidity,
        co2_conc=co2_conc,
        verbose=verbose)
    refraction_mag = refraction_atm['refraction_mag (arcsec)']

    num_waves = wavelength.size

    # get position angle:
    PA = coord.sun.P(input_times).degree

    parallactic_to_solar = refraction_atm['parallactic_angle (degrees)'] - PA

    # find the offsets due to atmospheric refraction:
    sfts_heliocent_ew = np.ones((num_times, num_waves))
    sfts_heliocent_ns = np.ones((num_times, num_waves))

    for wv in range(num_waves):
        sfts_heliocent_ew[:, wv] = (np.sin(np.radians(180
                                                      - parallactic_to_solar))
                                    * refraction_mag[:, wv])
        sfts_heliocent_ns[:, wv] = (np.cos(np.radians(180
                                                      - parallactic_to_solar))
                                    * refraction_mag[:, wv])

    if verbose == 1:
        print('\nPosition Angles in degrees: ', PA, '\n')
        for time, offsets in zip(input_times, sfts_heliocent_ew):
            print("East-West Offsets for Julian Date ", time, ": ", offsets)
        for time, offsets in zip(input_times, sfts_heliocent_ns):
            print("North-South Offsets for Julian Date ", time, ": ", offsets)

    offsets = {'East-West': sfts_heliocent_ew,
               'North-South': sfts_heliocent_ns}
    return(offsets, refraction_atm, coords)


# If executed as main program, run test case and compare with .sav file.

if __name__ == "__main__":

    do_haleakala = 1
    hatm = 11
    earth_radius = 6375.

    t_now = Time.now()
    t_now.format = 'iso'
    yyyy, mm, dd = t_now.value.split()[0].split('-')
    date_string = mm + '.' + dd + '.' + yyyy
    oname_pkl = 'refraction.calc.haleakala.' + date_string + '.pkl'
    oname_fits = 'coords.haleakala.fullyear.fits'

    haleakala = EarthLocation.of_site('dkist')

    # "average" atmospheric conditions on Haleakala
    temp = 11.*u.deg_C
    pressure = 71100*u.pascal
    humidity = 30.       # percent
    co2 = 380.      # ppm

    wavelengths = np.array([400, 525, 630, 700, 850, 1083, 1525])*u.nm
    num_waves = len(wavelengths)
    refrac = refractivity(wavelengths,
                          temp,
                          pressure,
                          humidity,
                          co2,
                          verbose=0)

    times_all_1d = (Time('2022-01-01 00:00:00', scale='utc')
                    + np.arange(1440*366) * u.min)
    times_all_1d.format = 'jd'

    if do_haleakala == 1:
        try:
            print("Reading coords from file " + oname_fits)
            hdul = fits.open(oname_fits)
            times_all_hka = Time(hdul[0].data.ravel(), format='jd', scale='utc')
            hours_ut_hka = ((times_all_hka[:1440].value
                             - times_all_1d[0].value) * 24) % 24
            alt_all = hdul[1].data.ravel()
            az_all = hdul[2].data.ravel()
            ha_all = hdul[3].data.ravel()
        except IOError:
            print("No coords file found. Please wait...")

            print("Calculating for Haleakalā")

            # Observatory: Haleakala
            longitude = -156.25*u.deg
            latitude = 20.71*u.deg
            altitude = 3055.*u.m

            # shift so that local noon is centered in array
            # array_shift_1d    = longitude / 360. * 1440.
            # but a half-day shift might be easier to understand/work with
            array_shift_1d = int(-longitude.value / 360. * 1440.) % 1440
            times_all_hka = times_all_1d[array_shift_1d:array_shift_1d + 365*1440]
            hours_ut_hka = ((times_all_hka[:1440].value
                             - times_all_1d[0].value) * 24) % 24

            print("Getting sun coordinates (RA/Dec)...")

            with erfa_astrom.set(ErfaAstromInterpolator(1440 * u.min)):
                sunpos = coordinates.get_sun(times_all_hka)
                print("Transforming to TETE (for proper HA calculation)")
                frame_TETE = TETE(obstime=times_all_hka, location=haleakala)
                sunpos_TETE = sunpos.transform_to(frame_TETE)
            local_sidereal = times_all_hka.sidereal_time('apparent', haleakala.lon)

            print("Transforming to Alt/Az...")

            # Get hour angle, altitude and azimuth
            # The rest of the program only uses the hour angle and altitude
            ha_all = ((local_sidereal - sunpos_TETE.ra).deg + 180) % 360 - 180
            with erfa_astrom.set(ErfaAstromInterpolator(1440 * u.min)):
                frame_obstime = AltAz(obstime=times_all_hka, location=haleakala)
                sunpos_altaz = sunpos.transform_to(frame_obstime)
            alt_all = sunpos_altaz.alt.deg
            az_all = sunpos_altaz.az.deg

            # save times, alt, az, ha in FITS file
            hdu = fits.PrimaryHDU(times_all_hka.reshape((365,1440)).value)
            hdu1 = fits.ImageHDU(alt_all.reshape((365,1440)))
            hdu2 = fits.ImageHDU(az_all.reshape((365,1440)))
            hdu3 = fits.ImageHDU(ha_all.reshape((365,1440)))
            hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
            hdul.writeto(oname_fits)

        times_all_hka_2d = np.reshape(times_all_hka, (365, 1440))
        alt_hka_2d = np.reshape(alt_all, (365, 1440))
        az_hka_2d = np.reshape(az_all, (365, 1440))
        ha_hka_2d = np.reshape(ha_all, (365, 1440))

        # below 4 degrees, the refraction calculations become unreliable
        altitude_mask = np.where(np.tile(alt_hka_2d.reshape(
            365, 1440, 1), (1, 1, num_waves)) >= 4, 1, 0)

        print("Calculating atmospheric dispersion...")

        dispersion_hka = np.zeros((365, 1440, num_waves))

        dispersion_temp = atmospheric_refraction(wavelengths,
                                                 times_all_hka,
                                                 alt_all,
                                                 az_all,
                                                 ha_all,
                                                 air_pressure=pressure,
                                                 air_temp=temp,
                                                 humidity=humidity,
                                                 co2_conc=co2,
                                                 observer_location=haleakala,
                                                 verbose=0)
        dispersion_1d = np.squeeze(
            dispersion_temp[0]['refraction_mag (arcsec)'])
        dispersion_hka = np.reshape(dispersion_1d, (365, 1440, num_waves))
        dispersion_hka = dispersion_hka * \
            altitude_mask + (1 - altitude_mask) * (-1)
        parang_hka = np.reshape(
            dispersion_temp[0]['parallactic_angle (degrees)'], (365, 1440))

        time_available_hka = np.zeros((365, 1440))
        time_available_hka_rot = np.zeros((365, 1440))
        time_available_hka_opt = np.zeros((365, 1440, 3))
        # DKIST presumed slit width
        slit_width_hka = 0.05

        tsize = 800
        toff = tsize / 2.
        tstart = (altitude_mask[:, :, 0] > 0).argmax(axis=1).min()

        disp_397 = dispersion_hka[:, :, np.argmin(np.abs(wavelengths
                                                         - 397*u.nm))]
        disp_854 = dispersion_hka[:, :, np.argmin(np.abs(wavelengths
                                                         - 854*u.nm))]
        parashift_hka_caII = disp_397 - disp_854

        print("Computing dispersion shifts...")

#        kernel_array = np.zeros((7, 7, 7))
#        kernel_array[3, :, 3] += 0.5
#        kernel_array[3, 1:-1, 3] += 0.5
#        kernel = CustomKernel(kernel_array)

        angle_lim = 90
        angle_delta = 1
        angle_extra = np.arange(-angle_lim,angle_lim,
                                angle_delta).reshape((1, 1, -1))
        num_angs = angle_extra.size
        zero_ang = int(np.argmin(np.abs(angle_extra)))
        angle_optimize = np.zeros((365, num_angs))
        parang_hka_big = parang_hka.reshape((365, 1440, 1))
        parashift_big = parashift_hka_caII.reshape((365, 1440, 1))

#        for i in range(365):
#            if ((i+1) % 20) == 0: print(i)
        for j in tqdm(range(tsize)):
            angle_curve = parang_hka_big[:, j+tstart:]
            dispersion_curve = np.clip(
                np.abs(parashift_big[:, j+tstart:]), None, 50)
            angle_tangent = np.clip(
                np.abs(np.sin((np.tile(angle_curve, (1, 1, num_angs))
                               - angle_curve[:, 0].reshape((365, 1, 1))
                               + angle_extra)*u.deg)), None, 2)
            perp_shift = angle_tangent * dispersion_curve
            altitude_curve = alt_hka_2d[:, j+tstart:].reshape((365, -1, 1))

#            perp_shift_conv = convolve(perp_shift,kernel)
            lim_slit_idx = (perp_shift >= slit_width_hka).argmax(axis=1)
            lim_alt_idx = (
                np.tile(altitude_curve, (1, 1, num_angs)) <= 3).argmax(axis=1)
            min_shift = np.minimum(lim_slit_idx, lim_alt_idx)
            time_available_hka[:, j+tstart] = min_shift[:, zero_ang]
            time_available_hka_rot[:, j+tstart] = angle_curve[np.arange(
                365), lim_slit_idx[:, zero_ang], 0]
            angle_optimize[:, :] = min_shift
            maxpos = np.argmax(angle_optimize, 1) - zero_ang
            max_time = np.max(angle_optimize, 1)
            time_available_hka_opt[:, j+tstart, 0] = max_time
            time_available_hka_opt[:, j+tstart, 1] = maxpos*angle_delta

        # save relevant arrays of parameters related to atmospheric dispersion

        output = {
            'times_all_hka_2d': times_all_hka_2d,
            'hours_ut_hka': hours_ut_hka,
            'wavelengths': wavelengths,
            'alt_hka_2d': alt_hka_2d,
            'az_hka_2d': az_hka_2d,
            'ha_hka_2d': ha_hka_2d,
            'parang_hka': parang_hka,
            'dispersion_hka': dispersion_hka,
            'time_available_hka': time_available_hka,
            'time_available_hka_rot': time_available_hka_rot,
            'time_available_hka_opt': time_available_hka_opt}
        with open(oname_pkl, 'wb') as f:
            pickle.dump(output, f)
        print("Results saved in file: " + oname_pkl)
