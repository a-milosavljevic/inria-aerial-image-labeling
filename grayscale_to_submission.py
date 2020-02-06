"""
This script is responsible for the preparation of final submission this can be submitted to the contest.
The prerequisites are grayscale predictions for test files located in tmp/submission_grayscale folder.
The threshold value is specified using cut_value parameter.
Since the contest requires mask files to be compressed using gdal_translate, GDAL must be installed and the path to
the gdal_translate must be specified.
"""
from data import *
import cv2 as cv
import os
import subprocess
import shutil


########################################################################################################################
# PARAMETERS
########################################################################################################################

cut_value = 0.45

grayscale_folder = os.path.join(tmp_folder, 'submission_grayscale')
submission_folder = os.path.join(tmp_folder, 'submission_{}'.format(cut_value))

gdal_translate = 'C:\\ms4w\\tools\\gdal-ogr\\gdal_translate'

if not os.path.exists(submission_folder):
    os.makedirs(submission_folder)


########################################################################################################################
# THRESHOLD IMAGES
########################################################################################################################

for filename in src_test_images:
    print('Processing image ' + filename)
    path = os.path.join(grayscale_folder, filename)
    path_out = os.path.join(submission_folder, filename)

    img = cv.imread(path, 0)

    ret, th = cv.threshold(img, round(255 * cut_value), 255, cv.THRESH_BINARY)

    cv.imwrite(path_out, th)


########################################################################################################################
# COMPRESS SUBMISSION
# Requires GDAL installed in the machine!
########################################################################################################################

command_template = '{} --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 {} {}'
for file in os.listdir(submission_folder):
    if file.endswith(".tif"):
        input_file = os.path.join(submission_folder, file)
        output_file = os.path.join(tmp_folder, file)
        command = command_template.format(gdal_translate, input_file, output_file)
        print(command)
        subprocess.call(command, shell=True)

        shutil.move(output_file, input_file)

shutil.make_archive(submission_folder, 'zip', submission_folder)
