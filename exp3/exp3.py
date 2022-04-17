import pandas as pd
import numpy as np
from scipy import stats
import glob
import warnings
warnings.filterwarnings("ignore")

def shannon_entropy(x):
    prob_energy = pow(x,2)/np.sum(pow(x,2))
    shannon = - np.sum(prob_energy*np.log2(prob_energy))
    return shannon

def lee(x):
    prob_energy = pow(x,2)/np.sum(pow(x,2))
    log_energy = np.sum(prob_energy*np.log(prob_energy))
    return log_energy

def individual_chars(path):
    data = pd.read_csv(path)
    for col in ['FZ -A2 ', 'CZ -A1 ','PZ -A2 ', 'BP2-REF']:
        data.pop(col)
    print(f"Individual Data for {path}\n")
    data_chars = pd.DataFrame({"Mean":data.mean(),"Median":data.median(),"Summation":data.sum(),"Variance":data.var(),"Standard Deviation":data.std(),"Shannon Entropy":shannon_entropy(data),"Log Energy Entropy":lee(data)})
    print(data_chars)


def hemisphere_chars(path):
    print(f"Hemisphere Data for {path}")
    data = pd.read_csv(path)
    for col in ['FZ -A2 ', 'CZ -A1 ','PZ -A2 ', 'BP2-REF']:
        data.pop(col)
    right_hemisphere = data[[i for i in data.columns if "A2" in i]].sum()
    left_hemisphere = data[[i for i in data.columns if "A1" in i]].sum()
    combined = pd.DataFrame({"Mean":[left_hemisphere.mean(),right_hemisphere.mean()],"Median":[left_hemisphere.median(),right_hemisphere.median()],"Summation":[left_hemisphere.sum(),right_hemisphere.sum()],"Variance":[left_hemisphere.var(),right_hemisphere.var()],"Standard Deviation":[left_hemisphere.std(),right_hemisphere.std()],"Shannon Entropy":[shannon_entropy(left_hemisphere),shannon_entropy(right_hemisphere)],"Log Energy Entropy":[lee(left_hemisphere),lee(right_hemisphere)]})
    combined.index = ["Left","Right"]
    combined = combined.transpose()
    print(combined)

def ecg_chars(path):
    print(f"ECG Data for {path}")
    data = pd.read_csv(path)
    data = data[["BP1-REF","BP2-REF"]]
    bp1 = data["BP1-REF"]
    bp2 = data["BP2-REF"]
    combined = pd.DataFrame({"Mean":[bp2.mean(),bp1.mean()],"Median":[bp2.median(),bp1.median()],"Summation":[bp2.sum(),bp1.sum()],"Variance":[bp2.var(),bp1.var()],"Standard Deviation":[bp2.std(),bp1.std()],"Shannon Entropy":[shannon_entropy(bp2),shannon_entropy(bp1)],"Log Energy Entropy":[lee(bp2),lee(bp1)]})
    combined.index = ["BP1","BP2"]
    combined = combined.transpose()
    print(combined)

for file in list(glob.glob("*"))[:2]:
    individual_chars(file)
    print("\n\n")
    
for file in list(glob.glob("*"))[:2]:
    hemisphere_chars(file)
    print("\n\n")
    
for file in list(glob.glob("*"))[:2]:
    ecg_chars(file)
    print("\n\n")
