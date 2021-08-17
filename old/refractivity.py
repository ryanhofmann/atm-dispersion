import numpy as np

def atmospheric_density(temperature=20, pressure_pa=100000, humidity=75, xc=380, force_xw=0, water_vapor=False, dry_air=False, verbose=0):
    TT_C = temperature
    TT_K = temperature + 273.15
    humidity_partial = humidity/100.
    if verbose >= 2:
        print("Temp: ", TT_C, "°C \nPressure: ", pressure_pa, " Pa \nHumidity: ", humidity, "% \nCO2: ", xc, " ppm")
        #there's probably a more technical, "easier" way to do that, but who cares? I like mine.
    
    #*************** Constants ****************
    #from Ciddor 1996, Appendix A:
    AA = 1.2378847*10**-5    #K^(-2)
    BB = -1.9121316*10**-2   #K^(-2)
    CC = 33.93711047         #
    DD = -6.3431645*10**3    #K
    
    alpha = np.float_(1.00062)#
    beta  = np.float_(3.14 * 10**-8)  #Pa^(-1)
    gamma = np.float(5.6 * 10**-7)    #°C^(-2)
    
    a0 = 1.58123*10**-6      #K Pa^(-1)
    a1 = -2.9331*10**-8      #Pa^(-1)
    a2 = 1.1043*10**-10      #K^(-1) Pa^(-1)
    b0 = 5.707*10**-6        #K Pa^(-1)
    b1 = -2.051*10**-8       #Pa^(-1)
    c0 = 1.9898*10**-4       #K Pa^(-1)
    c1 = -2.376*10**-6       #Pa^(-1)
    d  = 1.83*10**-11        #K^2 Pa^(-2)
    e  = -0.765*10**-8       #K^2 Pa^(-2)
    
    #from Ciddor 1995, Section 3
    #gas constant:
    R  = 8.314510            #J mol^(-1) K^(-1)
    #molar mass of water vapor:
    Mw = 0.018015            #kg/mol
    #molar mass of dry air containing a CO2 concentration of xc ppm:
    Malpha = (10**-3)*(28.9635 + (12.011*10**-6)*(xc-400)) 
    
    #***************End Constants*****************
    
    #saturation vapor pressure of water vapor in air at temperature T, from Ciddor 1996 Section 3:
    svp = np.exp(AA*TT_K**2 + BB*TT_K + CC + DD/TT_K)
    
    #enhancement factor of water vapor in air, whatever that is:
    f = alpha + beta*pressure_pa + gamma*TT_C**2
    
    if force_xw == 0:
        xw = f*humidity_partial*svp/pressure_pa
    else:
        xw=force_xw #molar fraction of water vapor
        
    #from Ciddor 1996 Appendix:    
    Z=1-(pressure_pa/TT_K)*(a0 + a1*TT_C + a2*TT_C**2 + (b0 + b1*TT_C)*xw + (c0 + c1*TT_C)*xw**2) + ((pressure_pa/TT_K)**2)*(d + e*xw**2)
    
    if water_vapor:
        density = (pressure_pa*Mw*xw/(Z*R*TT_K))
    elif dry_air:
        density = (pressure_pa * Malpha/(Z*R*TT_K))*(1 - xw)
    else:
        density = (pressure_pa * Malpha/(Z*R*TT_K))*(1 - xw*(1 - Mw/Malpha))
    
    if verbose >= 2:
        print("svp: ", svp, '\nf: ', f, '\nxw: ', xw, '\nZ: ', Z, '\ndensity: ', density)
    
    atmosphere_values = {'R':R, 'Z':Z, 'Ma':Malpha, 'Mw':Mw, 'svp':svp, 'f':f, 'density':density, 'TT_C':TT_C, 'TT_K':TT_K, 'pressure_pa':pressure_pa, 'humidity':humidity, 'xw':xw}

    return(density)


def refractivity(wavelength_nm=np.array([633.0]), temperature=20, pressure_pa=100000, humidity=75, xc=380, verbose=0):
    wavelength_mic = wavelength_nm/1000.
    
    #convert wavelengths in air to vacuum wavelengths [lambda(air) = lambda(vacuum)/n(air)]
    #using mean index of refraction of air = 1.00027
    wavelength_vac = wavelength_mic*1.00027
    wavenumber = 1/wavelength_vac
    
    #*****************  Constants  ******************
    #from Ciddor 1996, Appendix A
    #originally from Peck and Reeder 1962:
    k0 = 238.0185   #microns^(-2)
    k1 = 5792105.   #microns^(-2)
    k2 = 57.362     #microns^(-2)
    k3 = 167917.    #microns^(-2)
    
    #originally from Owens 1967:
    w0 = 295.235    #microns^(-2)
    w1 = 2.6422     #microns^(-2)
    w2 = -0.032380  #microns^(-4)
    w3 = 0.004028   #microns^(-6)
    #*************  End Constants  ******************
    
    #refractivity of air at 15°C, 101325 Pa, 0% humidity, and a fixed 450 ppm of CO2
    #from Ciddor 1996 Eq. 1:
    nas = (10**-8)*(k1/(k0 - wavenumber**2) + k3/(k2 - wavenumber**2)) + 1
    
    #refractivity of air at 15°C, 101325 Pa, 0% humidity, and a variable xc pmm of CO@
    #from Ciddor 1996 Eq. 2:
    naxs = (nas - 1)*(1 + (0.534*10**-6)*(xc - 450)) + 1
    
    #refractivity of water vapor at 20°C, 1333 Pa, 0% humidity
    #correction actor derived by Ciddor 1996 by fitting to measurements:
    cf = 1.022
    #from Ciddor 1996 Eq. 3:
    nws = (10**-8)*cf*(w0 + w1*wavenumber**2 + w2*wavenumber**4 + w3*wavenumber**6) + 1
    
    #density of dry air at standard conditions:
    density_axs = atmospheric_density(15, 101325, 0, xc, dry_air=True)
    #density of water vapor at standard conditions:
    density_ws  = atmospheric_density(20, 1333, 100, xc, force_xw=1)
    
    #density of dry air at input conditions:
    density_a = atmospheric_density(temperature, pressure_pa, humidity, xc, dry_air=True)
    #density of water vapor at input conditions:
    density_w = atmospheric_density(temperature, pressure_pa, humidity, xc, water_vapor=True)
    
    print("density a - ",density_a,density_axs,density_a/density_axs)
    print("density w - ",density_w,density_ws,density_w/density_ws)
    
    #from Ciddor 1996 Eq. 5:
    nprop_a = (density_a/density_axs)*(naxs - 1)
    nprop_w = (density_w/density_ws)*(nws - 1)
    nprop = nprop_a + nprop_w
    
    if verbose >= 1:
        print("n(axs): ", (naxs - 1)*10**8, "\nn(ws): ", (nws - 1)*10**8, "\nrho(a/axs): ", (density_a/density_axs), "\nrho(w/ws): ", (density_w/density_ws), "\nn(prop): ", nprop*10**8)
    if verbose >= 2:
        print("n(air): ", (density_a/density_axs)*(naxs - 1)*10**8, "\nn(water): ", (density_w/density_ws)*(nws - 1)*10**8)
    
    return(nprop)