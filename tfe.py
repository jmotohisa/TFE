#!/usr/bin/env python
# -*- coding: utf-8 -*-

# I-V Thermionic-field emission model
#
# PADOVANI, F. A., & Stratton, R. (1966).
# "Field and thermionic-field emission in Schottky barriers"
# Solid State Electronics, 9(7), 695–707. https://doi.org/10.1016/0038-1101(66)90097-9
#
# E. H. Rhoderick and R. H. Williams, ``Metal-Semiconductor Contacts''



import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import math

class parameters_tfe:

    def __init__(self,
                 temperature=300.,
                 ND=1e24,
                 eps_s=13.1,
                 ems=0.067,
                 phiB=1.,
                 Nc = 3e24,
                 ):

        self.temperature = temperature
        self.ND = ND
        self.eps_s = eps_s
        self.ems = ems
        self.phiB = phiB
        self.Astar = ARichardson(ems)
        self.Nc = Nc*(temperature/300)**1.5
        self.xi = math.log(self.Nc/ND)*(temperature*const.Boltzmann/const.elementary_charge)
        self.E00 = func_E00(ND,ems,eps_s)

    def output(self):
        print("temperature (K)=",self.temperature)
        print("Donor density (m^-3)=",self.ND)
        print("dielectric constant=",self.eps_s)
        print("effective mass=",self.ems)
        print("SBH=",self.phiB)
        print("Effective Rechardson Const=", self.Astar)
        print("Effective DOS of CB = ",self.Nc)
        print("Ec-Ef = ",self.xi)
        print("E00 (eV) = ", self.E00)

def EMS(ems):
    return ems*const.electron_mass

def EPSILON(eps):
    return eps*const.epsilon_0

# Thermionic emission model (TE)
#! effective Richardson constant
def ARichardson(ems):
    return 4*math.pi*(EMS(ems))*const.Boltzmann**2*const.elementary_charge/const.h**3


def Jte_forward(V,p):
    beta=const.elementary_charge/(p.temperature*const.Boltzmann)
    j0=(ARichardson(p.ems)*p.temperature**2*math.exp(-const.elementary_charge*p.phiB*beta))
    return j0*(math.exp(V*beta/p.n)-1)
    
#! 
# E_00: Eq. (13), (unit in eV)
def func_E00(ND,ems,eps_s):
    return const.hbar/2.0*math.sqrt(ND/(EMS(ems)*EPSILON(eps_s)))

def func_E0(E00,temp):
    return E00/math.tanh(const.elementary_charge*E00/(const.Boltzmann*temp))


# forward bias

# Saturation current
# Eq. (3.27a) in RW
def Jsf_TFE(Vf,p) :
    beta = const.elementary_charge//(p.temperature*const.Boltzmann)
    jm = p.Astar*p.temperature**2*math.exp(-p.xi*beta)
    Js = jm*const.elementary_charge*math.sqrt(math.pi*p.E00*(p.phiB-Vf-p.xi))
    Js = Js/(const.Boltzmann*p.temperature*math.cosh(p.E00*beta))
    Js = Js*math.exp(-(p.phiB-p.xi)/func_E0(p.E00,p.temperature))
    return Js

# Eq. (3.27) in RW
def Jf_TFE(Vf,p):
    beta=const.elementary_charge/(p.temperature*const.Boltzmann)
    E0=func_E0(p.E00,p.temperature)
    Js0=Jsf_TFE(Vf,p)
    return Js0*math.exp(Vf/E0)*(1-math.exp(-Vf*beta))


# reverse bias
# Eq.  (3.32) in RW
def Jsr_TFE(Vr,p) :
    beta=const.elementary_charge/(p.temperature*const.Boltzmann)
    Js = p.Astar*p.temperature*const.elementary_charge*math.sqrt(const.pi*p.E00)/const.Boltzmann
    Js = Js * math.sqrt(((-Vr-p.xi)+p.phiB/math.cosh(p.E00*beta)**2))
    Js = Js * math.exp(-p.phiB/func_E0(p.E00,p.temperature))
    return Js

def Jr_TFE(Vr,p) :
    beta=const.elementary_charge/(p.temperature*const.Boltzmann)
    Js0 = Jsr_TFE(Vr,p)
    return Js0*Jf_TFE(Vr,p)

if __name__ == '__main__':
    pGaN=parameters_tfe(ND=1e24,
                        ems=0.2,
                        phiB=0.9,
                        eps_s=8.9,
                        Nc=2.3e24)
    pGaN.output()
    Vf=np.linspace(0.01,0.8,endpoint=True)
    Vb=np.linspace(-5,0.01,endpoint=True)
    Jforward = np.empty_like(Vf)
    Jbackward = np.empty_like(Vb)
    for i,Vf0 in enumerate(Vf):
        Jforward[i] = abs(Jf_TFE(Vf0,pGaN))
    for i,Vb0 in enumerate(Vb):
        Jbackward[i] = abs(Jr_TFE(Vb0,pGaN))

    fig, ax = plt.subplots()
    ax.plot(Vf, Jforward, label='forward')
    ax.plot(Vb, Jbackward, label='backward')
    ax.set_yscale("log")
    # ax.set_ylim([1e-9, 1e-4])
    plt.ylabel('Current Density (A/m^2)')
    plt.xlabel('bias (V)')
    plt.legend(loc='best')
    plt.show()
        
