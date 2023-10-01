import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import sph_harm
from scipy.integrate import trapz
k_array = np.arange(0.005,3,0.005) # Sample for coefficients
E_array = np.arange(0,0.75,0.005) # Samples for angular distributions
phi_array = np.arange(-2*np.pi,2*np.pi,0.005)

suffix = "_600nm"

# Defines the asymmetry parameter from simulation
A = np.load("research_files/A" + f"{suffix}" + ".npy")
PES = np.load("research_files/PES" + f"{suffix}" + ".npy")

with open("research_files/coef_dict" + f"{suffix}"+".json", 'rb') as fp:
    coef = pickle.load(fp)
with open("research_files/coef_dict_organized" + f"{suffix}"+".json", 'rb') as fp:
    coef_organized = pickle.load(fp)
def Input_File_Reader(input_file = "research_files/input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters
def Index_Map(input_par):
    l_max = input_par["l_max"]
    m_max = input_par["m_max"]
    block_to_qn = {}
    qn_to_block = {}
    block = 0
    for m in range(0, m_max + 1):
            if m > 0:
                m_minus = -1*m
                for l in range(abs(m_minus), l_max + 1):
                    block_to_qn[block] = (l,m_minus)
                    qn_to_block[(l,m_minus)] = block
                    block += 1
            for l in range(m, l_max + 1):
                block_to_qn[block] = (l,m)
                qn_to_block[(l,m)] = block
                block += 1
    return  block_to_qn, qn_to_block

input_par = Input_File_Reader("research_files/input.json")
l_max = input_par["l_max"]


def lm_vals():
    lm_list = []
    for l in range(l_max+1):
        for m in range(-l,l+1):
            lm_list.append((l,m))
    lm_array = np.array(lm_list)
    np.save("research_files/lm_array.npy",lm_array)
    return
lm_array = np.load("research_files/lm_array.npy")

def closest(lst, k):
    return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-k))]
def peak_finder(PES,distance):
    indices_of_peaks = find_peaks(PES,distance = distance)[0]
    return indices_of_peaks,PES[indices_of_peaks],k_array[indices_of_peaks]
def plot_clm(l,m):
    return np.array([np.sqrt(coef[(l,m)][k][0]**2 + coef[(l,m)][k][1]**2) for k in k_array])
def plot_cks(k_s):
    lm_list = []
    c_lms = []
    for l,m in lm_array:
        k_closest = closest(k_array,k_s)
        real,imag = coef[(l,m)][k_closest]
        c_lms.append(np.sqrt(real**2 + imag**2))
        lm_list.append((l,m))
    return np.array(c_lms)
def best_lm_integrals(top):
    I_list = []
    for l,m in lm_array:
        c_of_k = plot_clm(l,m)
        I = trapz(c_of_k,k_array)
        I_list.append(I)
    I_array = np.array(I_list)
    top_lm_indices = np.argsort(I_array)[-top:]
    top_lm_vals = lm_array[top_lm_indices]
    return top_lm_vals
def best_lm_amp(k_s,top):
    cks = plot_cks(k_s)
    top_lm_indices = np.argsort(cks)[-top:]
    top_lm_values = lm_array[top_lm_indices]
    return top_lm_values
def Approx_PES(k_s,top):
    PES_approx = 0

    for l,m in best_lm_amp(k_s,top):
        PES_approx += coef[(l,m)][closest(k_array,k_s)][0]**2 + coef[(l,m)][closest(k_array,k_s)][1]**2
    return PES_approx
def K_Sphere(coef_dic,phi, theta,top_lm_values):
    theta, phi = np.meshgrid(theta, phi)
    out_going_wave = np.zeros(phi.shape, dtype=complex)
    for l,m in top_lm_values:
        c = coef_dic[(l,m)][0] + 1j*coef_dic[(l,m)][1]
        
        out_going_wave += c*sph_harm(m, l, phi, theta)
    return out_going_wave  

def approx_A_at_ATI(k_s,top):            
    theta = np.pi/2
    pad_value = np.zeros((phi_array.size))
    pad_value_rot = np.zeros((phi_array.size))
    
    top_lm_values = best_lm_amp(k_s,top)
    
    for phi_ind,phi in enumerate(phi_array):
        coef_dic = coef_organized[closest(list(coef_organized.keys()), k_s)]
    
        pad_value_k_phi =  np.abs(K_Sphere(coef_dic, phi, theta,top_lm_values))**2
        pad_value_k_phi_pi = np.abs(K_Sphere(coef_dic, phi+np.pi, theta,top_lm_values))**2
        
        pad_value[phi_ind] = pad_value_k_phi
        pad_value_rot[phi_ind] = pad_value_k_phi_pi
    numer = (pad_value - pad_value_rot)
    denom = (pad_value + pad_value_rot)

    A_approx = numer/denom
    
    
    
    index = list(E_array).index(closest(E_array,k_s**2 /2))
    A_exact = A[:,index]
    return A_exact,A_approx