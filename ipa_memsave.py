#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pymatgen.io.vasp.inputs import Poscar
import scipy.constants as scc
from scipy.integrate import quadrature
import sys
import matplotlib.pyplot as plt
import time
from multiprocessing import cpu_count, Pool, Manager
from functools import partial
def lorentz(x, x0, eps):
    return 1 / (x + x0 + 1j*eps) + 1 / (x - x0 - 1j*eps)

def phi(x):
    return np.exp(-1/(1-np.abs(x)**2)) / (x**2 + 7e-2)

# normalization constant so that the mollifier integrates to 1
#C = quadrature(phi,-1,1,tol=1e-10, rtol=1e-10)[0]
#C = 1

def phi_norm(x):
    return phi(x)/C

def mollifier(x,a):
    x2 = x*x
    return np.exp(-1/(1-x2)) / (x2 + a*a)

def moll_norm(x, x0, eps, a, C):
    eps1 = 1/eps
    o = np.zeros(len(x))
    if(not (np.min(x) > x0 + eps or np.max(x) < x0 - eps)):
        o += np.where(np.abs(x-x0) < eps, np.pi*mollifier((x-x0)*eps1 ,a)*eps1 / C, 0.0)
    if(not (np.min(x) > -x0 + eps or np.max(x) < -x0 - eps)):
        o -= np.where(np.abs(x+x0) < eps, np.pi*mollifier((x+x0)*eps1 ,a)*eps1 / C, 0.0)
    return o

def my_func(tt, energy, ep, a, C, gamma, idqueue):
    my_id = idqueue.get()
    
    p1 = np.zeros((3,len(energy)), dtype=np.complex128)
    p2 = np.zeros((3,len(energy)), dtype=np.complex128)
    i = 0
    j = 0
    j_tot = len(tt)
    tot_time = 0.0
    for t in tt:
        i += 1
        j += 1
        t1 = time.time()
        dE = t[3] - t[4]
        frac = t[0] / dE**2 * lorentz(energy, dE, gamma)
        frac2 = t[0] / dE**2 * moll_norm(energy.real, dE, ep, a, C)
        k1 = t[5]**2 + t[6]**2
        k2 = t[7]**2 + t[8]**2
        k3 = t[9]**2 + t[10]**2
        p1 += np.array([k1*frac, k2*frac, k3*frac])
        p2 += np.array([k1*frac2, k2*frac2, k3*frac2])
        tot_time += time.time() - t1
        if(my_id == 0 and i >> 10):
            i = 0
            print("Finished {:.3f}% - {:.2f}s".format(j/j_tot*100, tot_time), end='\r')

    if(my_id == 0):
        print("Finished {:.3f}% - {:.2f}s".format(100, tot_time))
    return p1, p2

class Transmatrix():
    def __init__(self, transmatrix):
        self.energy = None
        self.eps = None
        self.refractive = None
        self.absorption = None
        self.transmatrix = transmatrix
        self.nvb = int(self.transmatrix[0,1]-1)
        self.ncb = int(self.transmatrix[-1,1] - self.nvb)
        self.nkpt = int(len(self.transmatrix) / (self.nvb * self.ncb))

    def from_file(filename):
        return Transmatrix(pd.read_csv(filename, sep = "\s+", names = ["w", "c", "v", "ec", "ev", "repx", "impx", "repy", "impy", "repz", "impz" ]).values)

    
    def calculate_dielectric_function(self, emin, emax, de, gamma, g, cell_volume_bohr, gammaonly = False, nogamma = False):
        E_H = scc.physical_constants["Hartree energy in eV"][0]
        self.emin = emin
        self.emax = emax
        self.de = de
        self.gamma = gamma
        self.cell_volume = cell_volume_bohr
        self.energy = np.array(np.linspace(emin, emax, (emax-emin)/de+1), dtype=np.complex128)
        self.eps = np.zeros((3, len(self.energy)), dtype=np.complex128)
        self.imeps = np.zeros((3, len(self.energy)), float)

        length = len(self.transmatrix)
        C = quadrature(lambda x: mollifier(x,1/g) ,-1,1,tol=1e-10, rtol=1e-10, maxiter=1000)[0]
        ep = gamma * g
        a = 1 / g
        times = np.array([0.0, 0.0, 0.0])

        manager = Manager()
        idQueue = manager.Queue()
        
        cores = cpu_count()
        partitions = cores
        print("Running on {} cores!".format(cores))
        for i in range(cores):
            idQueue.put(i)

        if(gammaonly):
            data = self.transmatrix[:self.nvb*self.ncb]
        elif(nogamma):
            print("No gamma, starting at ", self.nvb * self.ncb)
            data = self.transmatrix[self.nvb*self.ncb:]
        else:
            data = self.transmatrix
        data_split = np.array_split(data, partitions)
        pool = Pool(cores)
        func = partial(my_func, energy = self.energy, ep=ep, a=a, C=C, gamma = gamma, idqueue=idQueue) 
        reduced = np.sum(list( pool.map(func, data_split)), axis=0)
        pool.close()
        pool.join()
        
        self.eps += reduced[0]
        self.imeps += reduced[1].real
        
        const = E_H**3 / cell_volume_bohr * 4 * np.pi # Prefactor dielectric function :D
        self.eps *= const
        self.imeps *= const
        self.eps += 1.0
        self.moleps = self.eps.copy()
        self.moleps.imag = self.imeps

    def calculate_refractive_index(self):
        self.refractive = np.sqrt(self.eps)
        self.molrefractive = np.sqrt(self.moleps)

    def calculate_absorption_coefficient(self):
        self.calculate_refractive_index()
        c = 1 / scc.physical_constants["fine-structure constant"][0]
        bohr_to_m = scc.physical_constants["Bohr radius"][0]
        E_H = scc.physical_constants["Hartree energy in eV"][0]

        self.absorption = self.energy / E_H * self.eps.imag / ( c * self.refractive.real) / bohr_to_m / 100
        self.molabsorption = self.energy / E_H * self.moleps.imag / ( c * self.molrefractive.real) / bohr_to_m / 100

    def get_energy(self):
        return self.energy

    def get_dielectric_function(self):
        return self.eps

    def get_mol_dielectric_function(self):
        return self.moleps

    def get_refractive_index(self):
        return self.refractive

    def get_mol_refractive_index(self):
        return self.molrefractive

    def get_absorption_coefficient(self):
        return self.absorption

    def get_mol_absorption_coefficient(self):
        return self.molabsorption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Parse dielectric function, refractive index and absorption coefficient.")
    parser.add_argument("-emin", type=float, dest="emin", help = "Lower energy boundary", required = True)
    parser.add_argument("-emax", type=float, dest="emax", help = "Upper energy boundary", required = True)
    parser.add_argument("-gamma", type=float, dest="gamma", help = "Spectrum broadening", required = True)
    parser.add_argument("-g", type=float, dest="g", help = "Fall off multiple", required = True)
    parser.add_argument("-gammaonly", action="store_true", dest="gammaonly", required = False, default = False)
    parser.add_argument("-nogamma", action="store_true", dest="nogamma", required = False, default = False)
    args = parser.parse_args()

    bohr_to_m = scc.physical_constants["Bohr radius"][0]
    bohr_to_cm = bohr_to_m * 1e2
    bohr_to_ang = bohr_to_m * 1e10
    tm = Transmatrix.from_file("Transmatrix")
    poscar = Poscar.from_file("CONTCAR").structure
    p = poscar.volume / bohr_to_ang**3 # Unit cell volume in Bohr radii

    tm.calculate_dielectric_function(args.emin, args.emax, args.gamma / 10.0, args.gamma, args.g,  p, args.gammaonly, args.nogamma)
    tm.calculate_absorption_coefficient()

    energy = tm.get_energy()
    diel = tm.get_dielectric_function()
    alpha = tm.get_absorption_coefficient()
    refractive = tm.get_refractive_index()

    moldiel = tm.get_mol_dielectric_function()
    molalpha = tm.get_mol_absorption_coefficient()
    molrefractive = tm.get_mol_refractive_index()

    str_add = "_{:.3g}_{:.3g}_{:.3g}".format(args.gamma, args.emin, args.emax)
    str_add_mol = "_{:.3g}_{:.3g}_{:.3g}_{:.3g}".format(args.gamma, args.g, args.emin, args.emax)

    if(args.gammaonly):
        str_add += "_gammaonly"
        str_add_mol += "_gammaonly"
    elif(args.nogamma):
        str_add += "_nogamma"
        str_add_mol += "_nogamma"
        

        
    np.savetxt(
        fname = "dielectric_function"+str_add+".dat",
        X = np.c_[ energy.real, diel.real.T, diel.imag.T ],
        header = """
            Dielectric Function over Energy (eV)
            Energy (eV)   Re(eps_x)   Re(eps_y)   Re(eps_z)   Im(eps_x)   Im(eps_y)   Im(eps_z)
        """
    )

    np.savetxt(
        fname = "absorption_coefficient"+str_add+".dat",
        X = np.c_[ energy.real, alpha.real.T ],
        header = """
            Absorption coefficient (cm^-1) over Energy (eV)
            Energy (eV)   Alpha_x   Alpha_y   Alpha_z
        """
    )

    np.savetxt(
        fname = "refractive_index"+str_add+".dat",
        X = np.c_[ energy.real, refractive.real.T, refractive.imag.T ],
        header = """
            Real and imaginary refractive index over Energy (eV)
            Energy (eV)   n_x   n_y   n_z   k_x   k_y   k_z
        """
    )

    np.savetxt(
        fname = "mollified_dielectric_function"+str_add_mol+".dat",
        X = np.c_[ energy.real, moldiel.real.T, moldiel.imag.T ],
        header = """
            Dielectric Function over Energy (eV)
            Energy (eV)   Re(eps_x)   Re(eps_y)   Re(eps_z)   Im(eps_x)   Im(eps_y)   Im(eps_z)
        """
    )

    np.savetxt(
        fname = "mollified_absorption_coefficient"+str_add_mol+".dat",
        X = np.c_[ energy.real, molalpha.real.T ],
        header = """
            Absorption coefficient (cm^-1) over Energy (eV)
            Energy (eV)   Alpha_x   Alpha_y   Alpha_z
        """
    )

    np.savetxt(
        fname = "mollified_refractive_index"+str_add_mol+".dat",
        X = np.c_[ energy.real, molrefractive.real.T, molrefractive.imag.T ],
        header = """
            Real and imaginary refractive index over Energy (eV)
            Energy (eV)   n_x   n_y   n_z   k_x   k_y   k_z
        """
    )
