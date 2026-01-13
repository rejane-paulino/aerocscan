# -*- mode: python -*-

"""

AerOC Scan is a package dedicated to download, filter, and correct Rrs spectra from AERONET-OC stations.
Author: Paulino (Nov-29-2024)

** Input Data:

*/ dest: output directory;
*/ level: AERONET-OC level (15 or 20);
*/ start: start date (yyyy-mm-dd);
*/ end: end date (yyyy-mm-dd);
*/ id: site id (or 'all');
*/ image_time: image time (hh:mm:ss);
*/ sensor_type: sensor type (S2A_MSI', 'S2B_MSI', S3A_OLCI', 'S3B_OLCI', 'L8_OLI', 'L9_OLI2', or 'PACE_OCI')

** Site id:

sites = {0: 'AAOT', 1: 'ARIAKE_TOWER_2', 2: 'Blyth_NOAH', 3: 'COVE_SEAPRISM', 4: 'Galata_Platform', 5: 'Grizzly_Bay', 6: 'Helsinki_Lighthouse',
         7: 'KAUST_Campus', 8: 'Lake_Okeechobee', 9: 'Lucinda', 10: 'PLOCAN_Tower', 11: 'San_Marco_Platform', 12: 'South_Greenbay', 13: 'USC_SEAPRISM_2',
         14: 'Zeebrugge-MOW1', 15: 'Abu_Al_Bukhoosh', 16: 'Bahia_Blanca', 17: 'Casablanca_Platform', 18: 'Frying_Pan_Tower', 19: 'Gloria', 20: 'Gustav_Dalen_Tower',
         21: 'Ieodo_Station', 22: 'Kemigawa_Offshore', 23: 'Lake_Okeechobee_N', 24: 'MVCO', 25: 'RdP-EsNM', 26: 'Section-7_Platform', 27: 'Thornton_C-power', 28: 'Venise',
         29: 'ARIAKE_TOWER', 30: 'Banana_River', 31: 'Chesapeake_Bay', 32: 'Gageocho_Station', 33: 'GOT_Seaprism', 34: 'HBOI', 35: 'Irbe_Lighthouse',
         36: 'Lake_Erie', 37: 'LISCO', 38: 'Palgrunden', 39: 'Sacramento_River', 40: 'Socheongcho', 41: 'USC_SEAPRISM', 42: 'WaveCIS_Site_CSI_6'}

** Outputs:

*/ 0-rawdata: data from AERONET-OC;
*/ 1-shapefile: site locations in .shp;
*/ 2-plots: median spectra from AERONET-OC Rrs;
*/ 3-rrs_median: AERONET-OC Rrs classified with OWTs;
*/ 4-X: Rrs spectra corrected by sensor.

"""

from aerocscan import AEROCSCAN

# Input:
dest = r'/Volumes/RSP/aeronet_oc/output'
level = '20' # 15 or 20
start = '2024-1-1' # yyyy-mm-dd
end = '2024-12-31' # yyyy-mm-dd
id = 'all' # 'all' - for all stations or specific int for station id
image_time = '13:00:00' # hh:mm:ss
sensor_type = 'S2A_MSI' # S2A_MSI', 'S2B_MSI', S3A_OLCI', 'S3B_OLCI', 'L8_OLI', 'L9_OLI2', or 'PACE_OCI'

# Without band correction:
aerocscan = AEROCSCAN(dest, level, id, start, end, image_time)
aerocscan.run()

# With band correction:
aerocscan = AEROCSCAN(dest, level, id, start, end, image_time, sensor_type)
aerocscan.run()
