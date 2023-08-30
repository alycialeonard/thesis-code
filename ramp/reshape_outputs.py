# reshape the output from a column to a big table
# input: column csv of power values with some header row
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import numpy as np
import pandas as pd

# the number of days that were originally simulated
n_days = 31

array = pd.read_csv("C:\\Users\\alyci\\OneDrive - Nexus365\\CCG\\SDEWES\\ramp results\\east_200_wealthequalized.csv").to_numpy().transpose()
reshape = np.reshape(array, (n_days, 1440)).transpose()
np.savetxt("C:\\Users\\alyci\\OneDrive - Nexus365\\CCG\\SDEWES\\ramp results\\east_200_wealthequalized-reshape.csv", reshape, delimiter=",")



