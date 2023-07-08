#INDEX:
#line 15:import thermophysical data
#line 27 import heat pipe data
#line 39 import input/output initial condition
#line 51 interpolation
#line 130 plot interpolation
#line 190 create constants
#line 244 initial conditions
#line 261 define power function
#line 281 model definition
#line 594 solve the model
#line 646 plot the solution
import pandas as pd

####################################################
#IMPORT THERMOPHYSICAL PROPERTIES
####################################################
filename_fluid = "water.xlsx" #Change here the fluid

thermophys_data = pd.ExcelFile(filename_fluid)
#print(type(thermophys_data))

thermophys_data = thermophys_data.parse("Sheet1") #overwrite to get a DataFrame
#print(type(thermophys_data))


####################################################
#IMPORT HEAT PIPE DATA
####################################################
filename_hp = "heat_pipe_specs.xlsx" #heat pipe data specification

hp_data = pd.ExcelFile(filename_hp)
#print(type(hp_data))

hp_data = hp_data.parse("Sheet1") #overwrite to get a DataFrame
#print(type(hp_data))


####################################################
#IMPORT INPUT/OUTPUT AND INITIAL CONDITION
####################################################
filename_par = "parameters.xlsx" #thermal parameters specification

parameters_data = pd.ExcelFile(filename_par)
#print(type(hp_data))

parameters_data = parameters_data.parse("Sheet1") #overwrite to get a DataFrame
#print(type(parameters_data))


####################################################
#INTERPOLATION
####################################################
import numpy as np
from numpy.polynomial import Polynomial
from scipy import interpolate


interpolation_method = "linear"


#extract data from DataFrame to arrays

temperature_for_fit = thermophys_data["Temperature (K)"] #get the dataframe slice
temperature_for_fit = temperature_for_fit.values #get only the values
temperature_for_fit = temperature_for_fit.tolist() #convert to list
temperature_for_fit = np.array([temperature_for_fit[0]-100] +  temperature_for_fit + [temperature_for_fit[-1]+100]) #extend and convert to an array


#poly_deg = 4 #degree of the polynomial


#Fit all the required parameters

#pressure_for_fit = thermophys_data["Saturation pressure (Pa)"].values
#p_sat = Polynomial.fit(temperature_for_fit, pressure_for_fit,poly_deg) #note that np.polyfit is now legacy
p_sat_for_fit = thermophys_data["Saturation pressure (Pa)"].values
p_sat_for_fit = p_sat_for_fit.tolist() #convert in list
p_sat_for_fit = np.array([p_sat_for_fit[0]] + p_sat_for_fit + [p_sat_for_fit[-1]]) #duplicate first and last elements

p_sat = interpolate.interp1d(temperature_for_fit, p_sat_for_fit,kind = interpolation_method) 


latentheat_for_fit = thermophys_data["Latent heat (kJ/kg)"].values
latentheat_for_fit = latentheat_for_fit.tolist() #convert in list
latentheat_for_fit = np.array([latentheat_for_fit[0]] + latentheat_for_fit + [latentheat_for_fit[-1]])*1e3 #duplicate first and last elements


#hlv_sat = Polynomial.fit(temperature_for_fit, latentheat_for_fit,poly_deg) 
hlv_sat = interpolate.interp1d(temperature_for_fit, latentheat_for_fit,kind = interpolation_method) 

liquiddensity_for_fit = thermophys_data["Liquid density (kg/m3)"].values.tolist()
liquiddensity_for_fit = np.array([liquiddensity_for_fit[0]] + liquiddensity_for_fit + [liquiddensity_for_fit[-1]])
#rhol_sat = Polynomial.fit(temperature_for_fit, liquiddensity_for_fit,poly_deg) 
rhol_sat =  interpolate.interp1d(temperature_for_fit, liquiddensity_for_fit,kind = interpolation_method) 

vapordensity_for_fit = thermophys_data["Vapor density (kg/m3)"].values.tolist()
vapordensity_for_fit = np.array([vapordensity_for_fit[0]] + vapordensity_for_fit + [vapordensity_for_fit[-1]])
#rhov_sat = Polynomial.fit(temperature_for_fit, vapordensity_for_fit,poly_deg) 
rhov_sat = interpolate.interp1d(temperature_for_fit, vapordensity_for_fit,kind= interpolation_method) 

liquidviscosity_for_fit = thermophys_data["Liquid viscosity (N-s/m2)"].values.tolist()
liquidviscosity_for_fit = np.array([liquidviscosity_for_fit[0]] + liquidviscosity_for_fit + [liquidviscosity_for_fit[-1]])
#mul_sat = Polynomial.fit(temperature_for_fit, liquidviscosity_for_fit,poly_deg) 
mul_sat = interpolate.interp1d(temperature_for_fit, liquidviscosity_for_fit,kind = interpolation_method) 

vaporviscosity_for_fit = thermophys_data["Vapor viscosity (N-s/m2)"].values.tolist()
vaporviscosity_for_fit = np.array([vaporviscosity_for_fit[0]] + vaporviscosity_for_fit + [vaporviscosity_for_fit[-1]])
#muv_sat = Polynomial.fit(temperature_for_fit, vaporviscosity_for_fit,poly_deg) 
muv_sat = interpolate.interp1d(temperature_for_fit, vaporviscosity_for_fit,kind=interpolation_method) 

liquidconductivity_for_fit = thermophys_data["Liquid thermal conductivity (W/m-K)"].values.tolist()
liquidconductivity_for_fit = np.array([liquidconductivity_for_fit[0]] + liquidconductivity_for_fit + [liquidconductivity_for_fit[-1]])
#kl_sat = Polynomial.fit(temperature_for_fit, liquidconductivity_for_fit,poly_deg) 
kl_sat = interpolate.interp1d(temperature_for_fit, liquidconductivity_for_fit,kind=interpolation_method)

liquidtension_for_fit = thermophys_data["Liquid surface tension (N/m)"].values.tolist()
liquidtension_for_fit = np.array([liquidtension_for_fit[0]] + liquidtension_for_fit + [liquidtension_for_fit[-1]])
#sigma_sat = Polynomial.fit(temperature_for_fit, liquidtension_for_fit,poly_deg) 
sigma_sat = interpolate.interp1d(temperature_for_fit, liquidtension_for_fit,kind = interpolation_method)

liquidcapacity_for_fit = (thermophys_data["Liquid specific heat (kJ/kg-K)"].values*1e3).tolist()
liquidcapacity_for_fit = np.array([liquidcapacity_for_fit[0]] + liquidcapacity_for_fit + [liquidcapacity_for_fit[-1]])
#cl_sat = Polynomial.fit(temperature_for_fit, liquidcapacity_for_fit,poly_deg) 
cl_sat = interpolate.interp1d(temperature_for_fit, liquidcapacity_for_fit,kind = interpolation_method) 


temperature_for_plot = np.linspace(temperature_for_fit[0],temperature_for_fit[-1], 1000)

####################################################
#PLOT THE INTERPOLATION
####################################################
import matplotlib.pyplot as plt


#create subplots

fig, axs = plt.subplots(4,2,sharex=True, figsize = (8,10))


axs[0,0].plot(temperature_for_plot,hlv_sat(temperature_for_plot))
axs[0,0].plot(temperature_for_fit,latentheat_for_fit, linestyle = "none", marker = "s")
axs[0,0].set_ylabel("latent heat (J/kg)")

axs[0,1].plot(temperature_for_plot,rhol_sat(temperature_for_plot))
axs[0,1].plot(temperature_for_fit,liquiddensity_for_fit, linestyle = "none", marker = "s")
axs[0,1].set_ylabel("$\u03C1_l$ (kg/m$^3$)")

axs[1,0].plot(temperature_for_plot,rhov_sat(temperature_for_plot))
axs[1,0].plot(temperature_for_fit,vapordensity_for_fit, linestyle = "none", marker = "s")
axs[1,0].set_ylabel("$\u03C1_v$ (kg/m$^3$)")

axs[1,1].plot(temperature_for_plot,mul_sat(temperature_for_plot))
axs[1,1].plot(temperature_for_fit,liquidviscosity_for_fit, linestyle = "none", marker = "s")
axs[1,1].set_ylabel("$\mu_l$ (Ns/m$^2$)")

axs[2,0].plot(temperature_for_plot,muv_sat(temperature_for_plot))
axs[2,0].plot(temperature_for_fit,vaporviscosity_for_fit, linestyle = "none", marker = "s")
axs[2,0].set_ylabel("$\mu_v$ (Ns/m$^2$)")

axs[2,1].plot(temperature_for_plot,kl_sat(temperature_for_plot))
axs[2,1].plot(temperature_for_fit,liquidconductivity_for_fit, linestyle = "none", marker = "s")
axs[2,1].set_ylabel("$k_v$ (Ns/m$^2$)")

axs[3,0].plot(temperature_for_plot,sigma_sat(temperature_for_plot))
axs[3,0].plot(temperature_for_fit,liquidtension_for_fit, linestyle = "none", marker = "s")
axs[3,0].set_ylabel("$\sigma$ (Ns/m$^2$)")
axs[3,0].set_xlabel("Temperature (K)")

axs[3,1].plot(temperature_for_plot,cl_sat(temperature_for_plot))
axs[3,1].plot(temperature_for_fit,liquidcapacity_for_fit, linestyle = "none", marker = "s")
axs[3,1].set_ylabel("$cp_l$ (J/kg-K)")
axs[3,1].set_xlabel("Temperature (K)")

axs[3,0].ticklabel_format(style="sci", axis = "y")
axs[3,1].ticklabel_format(style="sci", axis = "y")

axs[2,0].ticklabel_format(style="sci", axis = "y")
axs[2,1].ticklabel_format(style="sci", axis = "y")

axs[1,0].ticklabel_format(style="sci", axis = "y")
axs[1,1].ticklabel_format(style="sci", axis = "y")

axs[0,0].ticklabel_format(style="sci", axis = "y")
axs[0,1].ticklabel_format(style="sci", axis = "y")

fig.tight_layout() #adjust space between subplots, minimize overlap


####################################################
#CREATE THE CONSTANTS REQUIRED BY THE MODEL
####################################################
R = 8.314/(18e-3) #perfect gas reduced constant
gamma = 1.33      #cp/cv


#SOLID PROPERTIES
rhop = float(hp_data["pipe density (kg/m3)"].values)             #kg/m3
kp = float(hp_data["pipe thermal conductivity (W/m-K)"].values)  #W/mK
cp = float(hp_data["pipe heat capacity (J/kg-K)"].values)        #J/kg/K

rhow0 = float(hp_data["wick density (kg/m3)"].values)                #kg/m3
kw0 = float(hp_data["wick thermal conductivity (W/m-K)"].values)     #kg/m3
cw0 = float(hp_data["wick heat capacity (J/kg-K)"].values)           #kg/m3

#HP SPECS
dep = float(hp_data["pipe external diameter (m)"].values)
dip = float(hp_data["pipe internal diameter (m)"].values)
diw = float(hp_data["wick internal diameter (m)"].values)

print("outer pipe diameter (m): ", dep)
print("inner pipe diameter (m): ", dip)
print("inner wick diameter (m): ", diw)


Leva = float(hp_data["evaporator length (m)"].values)
La = float(hp_data["adiabatic length (m)"].values)
Lcond = float(hp_data["condenser length (m)"].values)

epsilon = float(hp_data["wick porosity"].values)
epsilon_eff = float(hp_data["wick effective porosity"].values)     #kg/m3
rc = float(hp_data["cavity radius (m)"].values)
K = float(hp_data["wick permeability"].values)

#compute cooling area in condenser
Af = Lcond*2*np.pi*(dep/2)+np.pi*(dep/2)**2  #condenser outer surface (cooling)
hf = float(parameters_data["htc (W/m2-K)"].values)


#VOLUMES
Vc = Lcond*np.pi*(diw**2)/4
Ve = Leva*np.pi*(diw**2)/4

#effective length
Leff = Leva/6 + La + Lcond/6

#INPUT POWER
Qin0 =float(parameters_data["input power (W)"].values)

#COOLING PART
Tf = float(parameters_data["cooling fluid temperature (K)"].values)


####################################################
#BUILD INITIAL CONDITIONS
####################################################
T0 = float(parameters_data["initial temperature"].values)

#INITIAL CONDITIONS
Mle_0 = rhol_sat(T0)*epsilon*Leva*np.pi*(dip**2 - diw**2)/4 #liquid mass in evaporator
Mlc_0 = rhol_sat(T0)*epsilon*Lcond*np.pi*(dip**2 - diw**2)/4 #liquid mass in adiabatic region
pve0 = float(p_sat(T0))#T0*R*rhov_sat(T0)#0.023360e5 pressure in evaporator
pvc0 = float(p_sat(T0))#T0*R*rhov_sat(T0)#0.023360e5 pressure in condenser
#T0*R*rhova
y0 = [T0,T0,T0,T0,T0,T0,0,pve0,pvc0,T0,T0,Mle_0,0*Mlc_0]

#float conversion is required otherwise the list is a mix between scalars and arrays

####################################################
#DEFINE THE INPUT POWER FUNCTION
####################################################
def Qin(t):
    return Qin0*(1-np.exp(-t/90))


#create a time vector for plot
tt = np.linspace(0,500,1000)

#create power vector for plot
Qin_plt = Qin(tt)

fig, ax = plt.subplots()
ax.plot(tt,Qin_plt)
ax.set_ylabel("Power (W)")
ax.set_xlabel("time (s)")
ax.set_title("Input power transient")

ax.grid(linestyle=":")


####################################################
#MATHEMATICAL MODEL DEFINITION
####################################################
def hp_model1(t,y):
    
    #for more clarity in the equation i save y vector in variables with clearer name
    Tpe = y[0] #pipe evaporator temperature
    Tpc = y[1] #pipe condenser temperature
    Twe = y[2] #wick evaporator temperature
    Twc = y[3] #wick condenser temperature
    Tpa = y[4] #pipe adiabatic temperature
    Twa = y[5] #wick adiabatic temperature
    mdotv = y[6] #vapor mass flow rate
    Pve = y[7] #vapor pressure evaporator region
    Pvc = y[8] #vapor pressure condenser region
    Tve = y[9] #vapor evaporator temperature
    Tvc = y[10] #vapor condenser temperature
    Mle = y[11] #liquid mass in evaporator
    Mlc = y[12] #liquid mass in condenser
  
    
    #get liquid termal conductivity from temperature
    kle = kl_sat(T0)#kl_sat(Twe)
    klc = kle#kl_sat(Twc)
    kla = kle#kl_sat(Twa)
    
    
    #thermophysical prop in the three regions
    mule = mul_sat(T0)#mul_sat(Twe)
    mula = mule#mul_sat(Twa)
    muva = muv_sat(T0)#(muv_sat(Tve) + muv_sat(Tvc))/2
    
    rhole = rhol_sat(T0)#rhol_sat(Twe)
    rhola = rhole#rhol_sat(Twa)
    rholc = rhola#rhol_sat(Twc)
    rhova = rhov_sat(T0)#(rhov_sat(Tve) + rhov_sat(Tvc))/2
    
    
    #wrapped screen wick effective thermal conductivity
    kwe = kle*((kle+kw0) - (1-epsilon_eff)*(kle-kw0))/((kle+kw0)+(1-epsilon_eff)*(kle-kw0))
    kwc = klc*((klc+kw0) - (1-epsilon_eff)*(klc-kw0))/((klc+kw0)+(1-epsilon_eff)*(klc-kw0))
    kwa = klc*((kla+kw0) - (1-epsilon_eff)*(kla-kw0))/((kla+kw0)+(1-epsilon_eff)*(kla-kw0))
    
    #sintered wick effective thermal conductivity
    #kwe = kw0*(2 + (kle/kw0) -2*epsilon_eff*(1- kle/kw0))/(2 + (kle/kw0)+epsilon_eff*(1-(kle/kw0)))
    #kwc = kw0*(2 + (klc/kw0) -2*epsilon_eff*(1- klc/kw0))/(2 + (klc/kw0)+epsilon_eff*(1-(klc/kw0)))
    #kwa = kw0*(2 + (kla/kw0) -2*epsilon_eff*(1- kla/kw0))/(2 + (kla/kw0)+epsilon_eff*(1-(kla/kw0)))
    
    
    #THERMAL RESISTANCES
    
    #radial resistance 
    R1pe = np.log(dep/(dip+dep)*2)/(2*np.pi*kp*Leva) #(dip+dep)/2 average diameter
    R2pe = np.log((dip+dep)/2/dip)/(2*np.pi*kp*Leva)
    R1we = np.log(dip/(diw+dip)*2)/(2*np.pi*kwe*Leva)
    R2we = np.log((diw+dip)/2/diw)/(2*np.pi*kwe*Leva)
    R1pc = np.log(dep/(dip+dep)*2)/(2*np.pi*kp*Lcond)
    R2pc = np.log((diw+dip)/2/diw)/(2*np.pi*kp*Lcond)
    R1wc = np.log(dip/(diw+dip)*2)/(2*np.pi*kwc*Lcond)#np.log(dep/(dip+dep)*2)/(2*np.pi*kwc*Lcond)
    R2wc = np.log((diw+dip)/2/diw)/(2*np.pi*kwc*Lcond)#np.log((diw+dip)/2/diw)/(2*np.pi*kwc*Lcond)
   

    #R2wa = np.log((diw+dip)/2/diw)/(2*np.pi*kwa*La)
    #R2wa = np.log((diw+dip)/2/diw)/(2*np.pi*kp*La)
    Rf = 1/hf/Af #external cooling resistance

    #axial resistances
    #which is the length?
    R1pa = (Leva+La)/2/kp/(np.pi*(dep**2-dip**2)/4)
    R2pa = (La+Lcond)/2/kp/(np.pi*(dep**2-dip**2)/4)
    R1wa = (Leva+La)/2/kwa/(np.pi*(dip**2-diw**2)/4)
    R2wa = (La+Lcond)/2/kwa/(np.pi*(dip**2-diw**2)/4)
    
    
    
    
    #specific heat in evaporator, adiabatic region, condenser
    cle = cl_sat(T0)#cl_sat(Twe) 
    clc = cle#cl_sat(Twc) 
    cla = cle#cl_sat(Twa) 

    
    #Thermal capacities
    Cpe  = rhop*cp*(Leva*np.pi*(dep**2-dip**2)/4)
    Cwe  = ((rhow0*cw0)*(1-epsilon)+(rhole*cle)*epsilon)*(Leva*np.pi*(dip**2-diw**2)/4)
    Cpa  = rhop*cp*(La*np.pi*(dep**2-dip**2)/4)
    Cwa  = ((rhow0*cw0)*(1-epsilon)+(rhola*cla)*epsilon)*(La*np.pi*(dip**2-diw**2)/4)
    Cpc  = rhop*cp*(Lcond*np.pi*(dep**2-dip**2)/4)
    Cwc  = ((rhow0*cw0)*(1-epsilon)+(rholc*clc)*epsilon)*(Lcond*np.pi*(dip**2-diw**2)/4)

    

    
    #vapor inductance and resistances
    Lva  = Leff/(np.pi*(diw/2)**2)
    Rle  = mule*Leva/(K*rhole*epsilon*np.pi*(dip**2-diw**2)/4)
    Rla  = mula/(K*rhola*epsilon*np.pi*(dip**2-diw**2)/4)
    Rva  = 8*muva*Leff/(rhova*np.pi*(diw/2)**4) 
    
    
    #thermophysical prop in ev, ad, cond
    hlve = hlv_sat(T0)#hlv_sat(Twe)
    hlvc = hlve#hlv_sat(Twc)
    
    sigmaa = sigma_sat(T0)#sigma_sat(Twa)
    sigmae = sigmaa#sigma_sat(Twe)
    sigmac = sigmaa#sigma_sat(Twc)
    sigma = np.mean([sigmaa,sigmae,sigmac])
    
    #UPDATE VAPOR CAPACITIES
    Cvc  = Vc/gamma/R/Tvc
    Cve  = Ve/gamma/R/Tve
 
    Qout = (Tpc-Tf)/(R1pc+Rf)

    
    #if Tpe >290:
    #    Tpe = 290
    
    if (Mle <= Mle_0) and (Mle > 0): #Mle_0 should be the mass with full filled wick
        f = -(Mle/Mle_0)+1
        f_masse = 1
    elif Mle>Mle_0: 
        f = (Pve-Pvc)/(2*sigma/rc)
        f_masse = 1
    else: 
        f = 0
        f_masse = 0
    

    #differential equations
    dydt = [
        Qin(t)/Cpe - (Tpe - Tpa)/R1pa/Cpe - (Tpe-Twe)/(R2pe+R1we)/Cpe,
        -(Tpc-Tpa)/R2pa/Cpc - (Tpc-Twc)/(R2pc+R1wc)/Cpc - Qout/Cpc,
        -(Twe -Twa)/R2wa/Cwe + (Tpe-Twe)/(R2pe+R1we)/Cwe - (Twe-Tve)/R2we/Cwe,
        -(Twc-Twa)/R2wa/Cwc + (Tpc-Twc)/(R2pc+R1wc)/Cwc - (Twc-Tvc)/R2wc/Cwc,
        -(Tpa-Tpe)/R1pa/Cpa - (Tpa-Tpc)/R2pa/Cpa,
        -(Twa-Twe)/R1wa/Cwa - (Twa-Twc)/R2wa/Cwa,
        Rva/Lva*mdotv + Pve/Lva - Pvc/Lva,
        f_masse*(Twe-Tve)/Cve/R2we/hlve - mdotv/Cve,
        mdotv/Cvc + (Twc-Tvc)/R2wc/hlvc/Cvc,
        (gamma-1)/gamma*Tve/Pve*(f_masse*(Twe-Tve)/Cve/R2we/hlve - mdotv/Cve), #check fmasse
        (gamma-1)/gamma*Tvc/Pvc*(mdotv/Cvc + (Twc-Tvc)/R2wc/hlvc/Cvc),
        #+ 2*sigma*f/Rva/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        #+ (Pvc-Pve)/Rva + 2*sigma*f/Rva/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        + (Pvc-Pve)/Rla + 2*sigma*f/Rla/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        #- 2*sigma*f/Rva/rc -(Twc-Tvc)/R2wc/hlvc]
        #- (Pvc-Pve)/Rva - 2*sigma*f/Rva/rc -(Twc-Tvc)/R2wc/hlvc]
        - (Pvc-Pve)/Rla - 2*sigma*f/Rla/rc -(Twc-Tvc)/R2wc/hlvc]
        
  
    return dydt

    

#model with temperature dependent parameters
def hp_modelTdependent(t,y):
    
    #for more clarity in the equation i save y vector in variables with clearer name
    Tpe = y[0] #pipe evaporator temperature
    Tpc = y[1] #pipe condenser temperature
    Twe = y[2] #wick evaporator temperature
    Twc = y[3] #wick condenser temperature
    Tpa = y[4] #pipe adiabatic temperature
    Twa = y[5] #wick adiabatic temperature
    mdotv = y[6] #vapor mass flow rate
    Pve = y[7] #vapor pressure evaporator region
    Pvc = y[8] #vapor pressure condenser region
    Tve = y[9] #vapor evaporator temperature
    Tvc = y[10] #vapor condenser temperature
    Mle = y[11] #liquid mass in evaporator
    Mlc = y[12] #liquid mass in condenser
  
    
    #get liquid termal conductivity from temperature
    kle = kl_sat(Twe)
    klc = kl_sat(Twc)
    kla = kl_sat(Twa)
    
    
    #thermophysical prop in the three regions
    mule = mul_sat(Twe)
    mula = mul_sat(Twa)
    muva = (muv_sat(Tve) + muv_sat(Tvc))/2
    
    rhole = rhol_sat(Twe)
    rhola = rhol_sat(Twa)
    rholc = rhol_sat(Twc)
    rhova = (rhov_sat(Tve) + rhov_sat(Tvc))/2
    
    
    #wrapped screen wick effective thermal conductivity
    kwe = kle*((kle+kw0) - (1-epsilon_eff)*(kle-kw0))/((kle+kw0)+(1-epsilon_eff)*(kle-kw0))
    kwc = klc*((klc+kw0) - (1-epsilon_eff)*(klc-kw0))/((klc+kw0)+(1-epsilon_eff)*(klc-kw0))
    kwa = klc*((kla+kw0) - (1-epsilon_eff)*(kla-kw0))/((kla+kw0)+(1-epsilon_eff)*(kla-kw0))
    
    #sintered wick effective thermal conductivity
    #kwe = kw0*(2 + (kle/kw0) -2*epsilon_eff*(1- kle/kw0))/(2 + (kle/kw0)+epsilon_eff*(1-(kle/kw0)))
    #kwc = kw0*(2 + (klc/kw0) -2*epsilon_eff*(1- klc/kw0))/(2 + (klc/kw0)+epsilon_eff*(1-(klc/kw0)))
    #kwa = kw0*(2 + (kla/kw0) -2*epsilon_eff*(1- kla/kw0))/(2 + (kla/kw0)+epsilon_eff*(1-(kla/kw0)))
    
    
    #THERMAL RESISTANCES
    
    #radial resistance 
    R1pe = np.log(dep/(dip+dep)*2)/(2*np.pi*kp*Leva) #(dip+dep)/2 average diameter
    R2pe = np.log((dip+dep)/2/dip)/(2*np.pi*kp*Leva)
    R1we = np.log(dip/(diw+dip)*2)/(2*np.pi*kwe*Leva)
    R2we = np.log((diw+dip)/2/diw)/(2*np.pi*kwe*Leva)
    R1pc = np.log(dep/(dip+dep)*2)/(2*np.pi*kp*Lcond)
    R2pc = np.log((diw+dip)/2/diw)/(2*np.pi*kp*Lcond)
    R1wc = np.log(dip/(diw+dip)*2)/(2*np.pi*kwc*Lcond)#np.log(dep/(dip+dep)*2)/(2*np.pi*kwc*Lcond)
    R2wc = np.log((diw+dip)/2/diw)/(2*np.pi*kwc*Lcond)#np.log((diw+dip)/2/diw)/(2*np.pi*kwc*Lcond)
   

    #R2wa = np.log((diw+dip)/2/diw)/(2*np.pi*kwa*La)
    #R2wa = np.log((diw+dip)/2/diw)/(2*np.pi*kp*La)
    Rf = 1/hf/Af #external cooling resistance

    #axial resistances
    #which is the length?
    R1pa = (Leva+La)/2/kp/(np.pi*(dep**2-dip**2)/4)
    R2pa = (La+Lcond)/2/kp/(np.pi*(dep**2-dip**2)/4)
    R1wa = (Leva+La)/2/kwa/(np.pi*(dip**2-diw**2)/4)
    R2wa = (La+Lcond)/2/kwa/(np.pi*(dip**2-diw**2)/4)
    
    
    
    
    #specific heat in evaporator, adiabatic region, condenser
    cle = cl_sat(Twe) 
    clc = cl_sat(Twc) 
    cla = cl_sat(Twa) 

    
    #Thermal capacities
    Cpe  = rhop*cp*(Leva*np.pi*(dep**2-dip**2)/4)
    Cwe  = ((rhow0*cw0)*(1-epsilon)+(rhole*cle)*epsilon)*(Leva*np.pi*(dip**2-diw**2)/4)
    Cpa  = rhop*cp*(La*np.pi*(dep**2-dip**2)/4)
    Cwa  = ((rhow0*cw0)*(1-epsilon)+(rhola*cla)*epsilon)*(La*np.pi*(dip**2-diw**2)/4)
    Cpc  = rhop*cp*(Lcond*np.pi*(dep**2-dip**2)/4)
    Cwc  = ((rhow0*cw0)*(1-epsilon)+(rholc*clc)*epsilon)*(Lcond*np.pi*(dip**2-diw**2)/4)

    

    
    #vapor inductance and resistances
    Lva  = Leff/(np.pi*(diw/2)**2)
    Rle  = mule*Leva/(K*rhole*epsilon*np.pi*(dip**2-diw**2)/4)
    Rla  = mula/(K*rhola*epsilon*np.pi*(dip**2-diw**2)/4)
    Rva  = 8*muva*Leff/(rhova*np.pi*(diw/2)**4) 
    
    
    #thermophysical prop in ev, ad, cond
    hlve = hlv_sat(Twe)
    hlvc = hlv_sat(Twc)
    
    sigmaa = sigma_sat(Twa)
    sigmae = sigma_sat(Twe)
    sigmac = sigma_sat(Twc)
    sigma = np.mean([sigmaa,sigmae,sigmac])
    
    #UPDATE VAPOR CAPACITIES
    Cvc  = Vc/gamma/R/Tvc
    Cve  = Ve/gamma/R/Tve


    
    Qout = (Tpc-Tf)/(R1pc+Rf)

    
    #if Tpe >290:
    #    Tpe = 290
    
    if (Mle <= Mle_0) and (Mle > 0): #Mle_0 should be the mass with full filled wick
        f = -(Mle/Mle_0)+1
        f_masse = 1
    elif Mle>Mle_0: 
        f = (Pve-Pvc)/(2*sigma/rc)
        f_masse = 1
    else: 
        f = 0
        f_masse = 0
    
    
    
    

    #differential equations
    dydt = [
        Qin(t)/Cpe - (Tpe - Tpa)/R1pa/Cpe - (Tpe-Twe)/(R2pe+R1we)/Cpe,
        -(Tpc-Tpa)/R2pa/Cpc - (Tpc-Twc)/(R2pc+R1wc)/Cpc - Qout/Cpc,
        -(Twe -Twa)/R2wa/Cwe + (Tpe-Twe)/(R2pe+R1we)/Cwe - (Twe-Tve)/R2we/Cwe,
        -(Twc-Twa)/R2wa/Cwc + (Tpc-Twc)/(R2pc+R1wc)/Cwc - (Twc-Tvc)/R2wc/Cwc,
        -(Tpa-Tpe)/R1pa/Cpa - (Tpa-Tpc)/R2pa/Cpa,
        -(Twa-Twe)/R1wa/Cwa - (Twa-Twc)/R2wa/Cwa,
        Rva/Lva*mdotv + Pve/Lva - Pvc/Lva,
        f_masse*(Twe-Tve)/Cve/R2we/hlve - mdotv/Cve,
        mdotv/Cvc + (Twc-Tvc)/R2wc/hlvc/Cvc,
        (gamma-1)/gamma*Tve/Pve*(f_masse*(Twe-Tve)/Cve/R2we/hlve - mdotv/Cve), #check fmasse
        (gamma-1)/gamma*Tvc/Pvc*(mdotv/Cvc + (Twc-Tvc)/R2wc/hlvc/Cvc),
        #+ 2*sigma*f/Rva/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        #+ (Pvc-Pve)/Rva + 2*sigma*f/Rva/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        + (Pvc-Pve)/Rla + 2*sigma*f/Rla/rc -f_masse*(Twe-Tve)/R2we/hlve, #check fmasse
        #- 2*sigma*f/Rva/rc -(Twc-Tvc)/R2wc/hlvc]
        #- (Pvc-Pve)/Rva - 2*sigma*f/Rva/rc -(Twc-Tvc)/R2wc/hlvc]
        - (Pvc-Pve)/Rla - 2*sigma*f/Rla/rc -(Twc-Tvc)/R2wc/hlvc]
        
        
    
    return dydt


####################################################
#SOLVE THE SYSTEM
####################################################
from scipy.integrate import solve_ivp
import time #required to measure the simulation time


#TIME SPAN
tt = [0,600]
#TIMESTEPS
Nsteps = 5000
tsteps = None#np.linspace(tt[0],tt[-1],Nsteps)


#SOLVE THE MODEL - Radau
st = time.time() #measure the starting time

solution = solve_ivp(hp_model1,t_span = tt, t_eval  = tsteps, y0 = y0 ,method='Radau')
et = time.time() #measure the final time

print("simulation time Radau:", round(et-st,4)) #print the simulation duration in seconds

#SAVE THE SOLUTION - Radau
t_rad = solution.t
y_rad = solution.y



#SOLVE THE MODEL - BDF

st = time.time()
solution = solve_ivp(hp_model1,t_span = tt, t_eval  = tsteps, y0 = y0 ,method='BDF')
et = time.time()

print("simulation time BDF:", round(et-st,4))

#SAVE THE SOLUTION - BDF
t_bdf = solution.t
y_bdf = solution.y

#SOLVE THE MODEL - LSODA

st = time.time()
solution = solve_ivp(hp_model1,t_span = tt, t_eval  = tsteps, y0 = y0 ,method='LSODA')
et = time.time()

print("simulation time LSODA:", round(et-st,4))

#SAVE THE SOLUTION - LSODA
t_lsoda = solution.t
y_lsoda = solution.y





####################################################
#PLOT THE SOLUTION
####################################################
#reference values
elgenk_times = [0,66,130,225,600]
elgenk_wallTempe = np.array([296,316,330,338,342])
elgenk_wallTempc = np.array([296,306,315,319,322])

#Create a subplot with 2 figures
fig, axs = plt.subplots(1,3,sharey=True,figsize = [10,5])

axs[0].plot(t_rad,y_rad[0],label="Tpe")
axs[0].plot(t_rad,y_rad[1],label="Tpc", linestyle="-.")
#axs[0].plot(t_rad,y_rad[2],label="Twe")
#axs[0].plot(t_rad,y_rad[3],label="Twc")
#axs[0].plot(t_rad,y_rad[4],label="Tpa")
#axs[0].plot(t_rad,y_rad[5],label="Twa")
axs[0].plot(elgenk_times, elgenk_wallTempe, label="Tpe experimental",color = 'blue', linestyle ="None", marker ="^")
axs[0].plot(elgenk_times, elgenk_wallTempc, label="Tpc experimental",color = 'C3', linestyle ="None", marker ="s")

axs[0].grid(linestyle=":")
axs[0].set_title("Radau solver")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Temperature (K)")

axs[1].plot(t_bdf,y_bdf[0],label="Tpe")
axs[1].plot(t_bdf,y_bdf[1],label="Tpc", linestyle="-.")
#axs[1].plot(t_bdf,y_bdf[2],label="Twe")
#axs[1].plot(t_bdf,y_bdf[3],label="Twc")
#axs[1].plot(t_bdf,y_bdf[4],label="Tpa")
#axs[1].plot(t_bdf,y_bdf[5],label="Twa")
axs[1].plot(elgenk_times, elgenk_wallTempe, label="Tpe experimental",color = 'blue', linestyle ="None", marker ="^")
axs[1].plot(elgenk_times, elgenk_wallTempc, label="Tpc experimental",color = 'C3', linestyle ="None", marker ="s")


axs[1].grid(linestyle=":")
axs[1].set_title("BDF solver")
axs[1].set_xlabel("Time (s)")


axs[2].plot(t_lsoda,y_lsoda[0],label="Tpe")
axs[2].plot(t_lsoda,y_lsoda[1],label="Tpc", linestyle="-.")
#axs[2].plot(t_lsoda,y_lsoda[2],label="Twe")
#axs[2].plot(t_lsoda,y_lsoda[3],label="Twc")
#axs[2].plot(t_lsoda,y_lsoda[4],label="Tpa")
#axs[2].plot(t_lsoda,y_lsoda[5],label="Twa")
axs[2].plot(elgenk_times, elgenk_wallTempe, label="Tpe experimental", color = 'blue', linestyle ="None", marker ="^")
axs[2].plot(elgenk_times, elgenk_wallTempc, label="Tpc experimental",color = 'C3', linestyle ="None", marker ="s")


axs[2].grid(linestyle=":")
axs[2].set_title("LSODA solver")
axs[2].set_xlabel("Time (s)")



axs[0].legend(loc='lower right')
axs[1].legend(loc='lower right')
axs[2].legend(loc='lower right')

plt.tight_layout()

#PLOT PRESSURES
fig, axs = plt.subplots(1,3,sharey=True,figsize = [8,4])

axs[0].plot(t_rad,y_rad[7], label="Pve")
axs[0].plot(t_rad,y_rad[8], label="Pvc", linestyle="-.")
axs[0].set_title("Radau solver")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Pressure (Pa)")

axs[1].plot(t_bdf,y_bdf[7], label="Pve")
axs[1].plot(t_bdf,y_bdf[8], label="Pvc", linestyle="-.")
axs[1].set_title("BDF solver")
axs[1].set_xlabel("Time (s)")

axs[2].plot(t_lsoda,y_lsoda[7], label="Pve")
axs[2].plot(t_lsoda,y_lsoda[8], label="Pvc", linestyle="-.")
axs[2].set_title("LSODA solver")
axs[2].set_xlabel("Time (s)")

axs[0].grid(linestyle=":")
axs[1].grid(linestyle=":")
axs[2].grid(linestyle=":")

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.tight_layout()

#PLOT MASSES
fig, axs = plt.subplots(1,3,sharey=True,figsize = [8,4])

axs[0].plot(t_rad,y_rad[11], label="Mle")
axs[0].plot(t_rad,y_rad[12], label="Mlc", linestyle="-.")
axs[0].set_title("Radau solver")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Liquid mass (kg)")

axs[1].plot(t_bdf,y_bdf[11], label="Mle")
axs[1].plot(t_bdf,y_bdf[12], label="Mlc", linestyle="-.")
axs[1].set_title("BDF solver")
axs[1].set_xlabel("Time (s)")

axs[2].plot(t_lsoda,y_lsoda[11], label="Mle")
axs[2].plot(t_lsoda,y_lsoda[12], label="Mlc", linestyle="-.")
axs[2].set_title("BLSODA solver")
axs[2].set_xlabel("Time (s)")

axs[0].grid(linestyle=":")
axs[1].grid(linestyle=":")
axs[2].grid(linestyle=":")

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.tight_layout()
