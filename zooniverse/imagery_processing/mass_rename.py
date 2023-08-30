# Rename a bunch of images in a directory
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import glob
import os


def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename,
                  os.path.join(dir, titlePattern % title + ext))


rename(r'C:\Users\alyci\PycharmProjects\gbdx\DATA\google_earth\west_ngosini_samples\small_crop', r'*.jpg', r'westngosini_%s')
