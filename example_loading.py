'''

This is an example of loading a tomogram as a numpy array.
The example uses "mrcfile" package, which is installable
	by PIP: pip install mrcfile
	by Conda: conda install --channel conda-forge mrcfile

mrcfile documentation can be found here: https://mrcfile.readthedocs.io/

Contact us if you have questions at i.gubins@uu.nl

'''

from __future__ import print_function # for python 2 compatibility of prints
import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos

# permissive=True is neccessary for our .mrc tomos - we don't populate header correctly
with mrcfile.open('0/reconstruction_model_0.mrc', permissive=True) as tomo0: 
	
	# the tomo data is now accessible via .data
	full_tomogram = tomo0.data

	# you can also apply numpy/python slicing to the data
	center_crop = tomo0.data[50:-50, 50:-50, 50:-50]

	# .data is a numpy array with all numpy features
	print('Shape of the tomogram:', full_tomogram.shape)
	print('Size of the tomogram:', full_tomogram.nbytes / float(1000**3), 'GB')
