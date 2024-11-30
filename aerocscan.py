# -*- mode: python -*-

import shutil
import os

import toolbox as tool


class AEROCSCAN:

    def __init__(self, dest: str, level: str, id: str, start: str, end: str, local_time: str, sensor_type=None):

        self.dest = dest
        self.level = level
        self.id = id
        self.start = start
        self.end = end
        self.local_time = local_time
        self.sensor_type = sensor_type

    def run(self):

        # Creates the 'tempdir':
        dirout = self.dest + '/tempdir'
        os.makedirs(dirout, exist_ok=True)
        # Download - AERONET-OC:
        tool.download_aerOC(self.dest, self.start, self.end, self.level, self.id)
        # Filters the timeframe:
        tool.filtering_timeframe(self.dest + '/0-rawdata', self.start, self.end, self.local_time, dirout)
        # Filters the parameters:
        tool.filtering_parameters(dirout + '/timeframe', dirout)
        # Calculates the rrs:
        tool.rrs(dirout + '/parameters', self.dest)
        # Classifies the OWTs:
        tool.owts(dirout + '/rrs_median', self.dest)

        if self.sensor_type != None:
            # Corrects the AERONET-OC bands according the sensor type:
            tool.shiftband_corr(self.dest + '/3-rrs_median', self.sensor_type, self.dest)
        else:
            None

        tool.shapefile(self.dest + '/3-rrs_median', self.dest)
        shutil.rmtree(dirout)
