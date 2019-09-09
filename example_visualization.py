'''

This is an example of parsing ground truth.
The script goes through all ground truth entries,
and visualizes P random subtomograms in ground truth and reconstruction of model N.

The example uses "mrcfile" package, which is installable
	by PIP: pip install mrcfile
	by Conda: conda install --channel conda-forge mrcfile

mrcfile documentation can be found here: https://mrcfile.readthedocs.io/

Contact us if you have questions at i.gubins@uu.nl

'''

from __future__ import print_function # for python 2 compatibility of prints
import numpy as np
import mrcfile as mrc
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos

# N should be from 0 to 8 (9th tomo doesn't have ground truth)
# N defines which model information to load and parse
N = 4
# P should be any positive integer
# P defines how many particles to plot
P = 5

# read ground truth model
with mrc.open(r'{}/grandmodel_{}.mrc'.format(N, N), permissive=True) as f:
	gt_data = f.data

# read reconstructed model
with mrc.open(r'{}/reconstruction_model_{}.mrc'.format(N, N), permissive=True) as f:
	exp_data = f.data

# parse ground truth text file
locations = []
with open(r'{}/particle_locations_model_{}.txt'.format(N, N), 'rU') as f:
	for line in f:
		pdb_id, Z, Y, X, rot_X, rot_Y, rot_Z = line.rstrip('\n').split()
		locations.append((pdb_id, int(Z), int(Y), int(X)))

# create a figure
fig, ax = plt.subplots(P, 2, sharex=True, sharey=True)
ax[0, 0].set_title('Ground truth')
ax[0, 1].set_title('Reconstruction')

# show in figure 10 random 64x64 slices
i = 0
np.random.shuffle(locations)
for pdb_id, Z, Y, X in locations:
	# extract central slice of a subtomogram
	subtomo_gt = gt_data[Z, Y-32:Y+32, X-32:X+32]
	subtomo = exp_data[Z+156, Y-32:Y+32, X-32:X+32]

	# for a prettier demo, use only 64x64 slices
	if (subtomo.shape != (64, 64) or subtomo_gt.shape != (64, 64)):
		continue

	# plot
	ax[i, 0].imshow(subtomo_gt)
	ax[i, 1].imshow(subtomo)

	# only first 10
	i += 1
	if i > P-1:
		break

plt.show()