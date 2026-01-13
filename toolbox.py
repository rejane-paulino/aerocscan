
import os
import wget
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib import pyplot as plt


def download_aerOC(dest: str, start: str, end: str, level: str, id) -> None:
    """
    It downloads the AERONET-OC data considering a single or all stations.
    :param dest: folder to save all files;
    :param start: start date (yyyy-mm-day);
    :param end: end date (yyyy-mm-day);
    :param level: product level - 20 (2.0) or 15 (1.5);
    :param id: station id;
    """

    sites = {0: 'AAOT', 1: 'ARIAKE_TOWER_2', 2: 'Blyth_NOAH', 3: 'COVE_SEAPRISM', 4: 'Galata_Platform', 5: 'Grizzly_Bay', 6: 'Helsinki_Lighthouse',
             7: 'KAUST_Campus', 8: 'Lake_Okeechobee', 9: 'Lucinda', 10: 'PLOCAN_Tower', 11: 'San_Marco_Platform', 12: 'South_Greenbay', 13: 'USC_SEAPRISM_2',
             14: 'Zeebrugge-MOW1', 15: 'Abu_Al_Bukhoosh', 16: 'Bahia_Blanca', 17: 'Casablanca_Platform', 18: 'Frying_Pan_Tower', 19: 'Gloria', 20: 'Gustav_Dalen_Tower',
             21: 'Ieodo_Station', 22: 'Kemigawa_Offshore', 23: 'Lake_Okeechobee_N', 24: 'MVCO', 25: 'RdP-EsNM', 26: 'Section-7_Platform', 27: 'Thornton_C-power', 28: 'Venise',
             29: 'ARIAKE_TOWER', 30: 'Banana_River', 31: 'Chesapeake_Bay', 32: 'Gageocho_Station', 33: 'GOT_Seaprism', 34: 'HBOI', 35: 'Irbe_Lighthouse',
             36: 'Lake_Erie', 37: 'LISCO', 38: 'Palgrunden', 39: 'Sacramento_River', 40: 'Socheongcho', 41: 'USC_SEAPRISM', 42: 'WaveCIS_Site_CSI_6'}

    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    # Creating a directory:
    dirout = dest + '/0-rawdata'
    os.makedirs(dirout, exist_ok=True)

    if id == 'all':
        for site in sites:
            station_id = sites[site]
            filename_out = start.strftime("%Y%m%d") + '_' + end.strftime("%Y%m%d") + '_' + station_id + '.LWN_lev' + level

            url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site='+station_id+\
                '&year='+str(start.year)+'&month='+str(start.month)+'&day='+str(start.month)+\
                '&year2='+str(end.year)+'&month2='+str(end.month)+'&day2='+str(end.month)+\
                '&LWN'+str(level)+'=1&AVG=10&if_no_html=1'

            wget.download(url, out=os.sep.join([dirout, filename_out]))
    else:
        station_id = sites[id]
        filename_out = start.strftime("%Y%m%d") + '_' + end.strftime("%Y%m%d") + '_' + station_id + '.LWN_lev' + level

        url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site='+station_id+\
            '&year='+str(start.year)+'&month='+str(start.month)+'&day='+str(start.month)+\
            '&year2='+str(end.year)+'&month2='+str(end.month)+'&day2='+str(end.month)+\
            '&LWN'+str(level)+'=1&AVG=10&if_no_html=1'

        wget.download(url, out=os.sep.join([dirout, filename_out]))

    return None


def filtering_timeframe(in_path: str, start: str, end: str, image_time: str, dest: str) -> None:
    """
    Filters the AERONET-OC according the window time (+/- 1.5 hours).
    """

    # Creating a directory:
    dirout = dest + '/timeframe'
    os.makedirs(dirout, exist_ok=True)

    for file in os.listdir(in_path):
        try:
            # Opens the files:
            data = pd.read_csv(in_path + '/' + file, delimiter=',', skiprows=5, engine='python')

            # Combines date and time columns into a single datetime column:
            data['Datetime'] = pd.to_datetime(data['Date(dd-mm-yyyy)'] + ' ' + data['Time(hh:mm:ss)'], format='%d:%m:%Y %H:%M:%S')

            # Filters the dataset considering +/- 1.5-hours timeframe:
            datetime_start = datetime.strptime(f"{start} {image_time}", '%Y-%m-%d %H:%M:%S')
            datetime_end = datetime.strptime(f"{end} {image_time}", '%Y-%m-%d %H:%M:%S')
            timeframe = timedelta(hours=1.5)  # timeframe
            datetime_start = (datetime_start - timeframe).strftime('%Y-%m-%d %H:%M:%S')
            datetime_end = (datetime_end + timeframe).strftime('%Y-%m-%d %H:%M:%S')

            filtered = data[(data['Datetime'].dt.time >= pd.to_datetime(datetime_start).time()) &
                            (data['Datetime'].dt.time <= pd.to_datetime(datetime_end).time())]


            if not filtered.empty:
                filtered.to_csv(dirout + '/' + file + '.csv', sep=',')
            else:
                None

        except:
            pass

    return None


def filtering_parameters(in_path: str, dest: str) -> None:
    """
    Filters the AERONET-OC according the atmospheric and radiometric data.
    """

    # Creating a directory:
    dirout = dest + '/parameters'
    os.makedirs(dirout, exist_ok=True)

    for file in os.listdir(in_path):
        try:
            data = pd.read_csv(in_path + '/' + file, delimiter=',', engine='python')

            filtered = data.filter(['Date(dd-mm-yyyy)', 'Total_Precipitable_Water(cm)', 'Total_Ozone(Du)',
                                    'Aerosol_Optical_Depth[340nm]', 'Aerosol_Optical_Depth[380nm]', 'Aerosol_Optical_Depth[400nm]', 'Aerosol_Optical_Depth[412nm]', 'Aerosol_Optical_Depth[440nm]', 'Aerosol_Optical_Depth[443nm]',
                                    'Aerosol_Optical_Depth[490nm]', 'Aerosol_Optical_Depth[500nm]', 'Aerosol_Optical_Depth[510nm]', 'Aerosol_Optical_Depth[531nm]', 'Aerosol_Optical_Depth[532nm]', 'Aerosol_Optical_Depth[551nm]',
                                    'Aerosol_Optical_Depth[555nm]', 'Aerosol_Optical_Depth[560nm]', 'Aerosol_Optical_Depth[620nm]', 'Aerosol_Optical_Depth[667nm]', 'Aerosol_Optical_Depth[675nm]', 'Aerosol_Optical_Depth[681nm]',
                                    'Aerosol_Optical_Depth[709nm]', 'Aerosol_Optical_Depth[779nm]', 'Aerosol_Optical_Depth[865nm]', 'Aerosol_Optical_Depth[870nm]', 'Aerosol_Optical_Depth[1020nm]', 'Lwn_IOP[340nm]', 'Lwn_IOP[380nm]',
                                    'Lwn_IOP[400nm]', 'Lwn_IOP[412nm]', 'Lwn_IOP[440nm]', 'Lwn_IOP[443nm]', 'Lwn_IOP[490nm]', 'Lwn_IOP[500nm]', 'Lwn_IOP[510nm]', 'Lwn_IOP[531nm]', 'Lwn_IOP[532nm]', 'Lwn_IOP[551nm]', 'Lwn_IOP[555nm]',
                                    'Lwn_IOP[560nm]', 'Lwn_IOP[620nm]', 'Lwn_IOP[667nm]', 'Lwn_IOP[675nm]', 'Lwn_IOP[681nm]', 'Lwn_IOP[709nm]', 'Lwn_IOP[779nm]', 'Lwn_IOP[865nm]', 'Lwn_IOP[870nm]',
                                    'Chlorophyll-a', 'Site_Elevation(m)'])

            unique_levels = filtered['Date(dd-mm-yyyy)'].unique()
            bands = ['340', '380', '400', '412', '440', '443', '490', '500', '510', '531', '532', '551', '555', '560', '620', '667', '675', '681', '709', '779', '865', '870', '1020']
            list_data = []
            # Atmospheric parameters:
            for i in unique_levels:
                filter_data = filtered.loc[filtered['Date(dd-mm-yyyy)'] == i]
                # AOD param - based on Eck et al. (1999):
                filter_aod = filter_data.filter(['Aerosol_Optical_Depth[340nm]', 'Aerosol_Optical_Depth[380nm]', 'Aerosol_Optical_Depth[400nm]', 'Aerosol_Optical_Depth[412nm]', 'Aerosol_Optical_Depth[440nm]', 'Aerosol_Optical_Depth[443nm]',
                                              'Aerosol_Optical_Depth[490nm]', 'Aerosol_Optical_Depth[500nm]', 'Aerosol_Optical_Depth[510nm]', 'Aerosol_Optical_Depth[531nm]', 'Aerosol_Optical_Depth[532nm]', 'Aerosol_Optical_Depth[551nm]',
                                              'Aerosol_Optical_Depth[555nm]', 'Aerosol_Optical_Depth[560nm]', 'Aerosol_Optical_Depth[620nm]', 'Aerosol_Optical_Depth[667nm]', 'Aerosol_Optical_Depth[675nm]', 'Aerosol_Optical_Depth[681nm]',
                                              'Aerosol_Optical_Depth[709nm]', 'Aerosol_Optical_Depth[779nm]', 'Aerosol_Optical_Depth[865nm]', 'Aerosol_Optical_Depth[870nm]', 'Aerosol_Optical_Depth[1020nm]'])
                def model_function(lambida, b1, b2, b3):
                    ln_lambda = np.log(lambida)
                    return b1 * (ln_lambda ** 2) + b2 * ln_lambda + b3
                aod_l = []
                for parm in range(0, len(filter_aod)):
                    array_r = np.array(filter_aod.reset_index().loc[parm][1:])
                    bands_v = np.where(array_r > 0, bands, '0') # mapping the spectral bands
                    bands_v = np.array([int(k) for k in bands_v if k != '0'])
                    array_v = array_r[array_r > 0] # masking NaN data
                    ln_aod_values = np.log(array_v)
                    popt, pcov = curve_fit(model_function, bands_v, ln_aod_values)
                    b1_fitted, b2_fitted, b3_fitted = popt
                    lambda_550 = 550
                    aod_550_fitted = np.exp(model_function(lambda_550, b1_fitted, b2_fitted, b3_fitted))
                    aod_l.append(aod_550_fitted)
                aod_value = np.mean(aod_l)
                ozone_value = filter_data['Total_Ozone(Du)'].mean() / 1000 # cm-atm
                water_vapor = filter_data['Total_Precipitable_Water(cm)'].mean() # g/cm2
                altitude = filter_data['Site_Elevation(m)'].mean() / 1000 # km
                chl = filter_data['Chlorophyll-a'].mean()
                filter_rad = filter_data.filter(['Date(dd-mm-yyyy)', 'Lwn_IOP[340nm]', 'Lwn_IOP[380nm]','Lwn_IOP[400nm]',
                                            'Lwn_IOP[412nm]','Lwn_IOP[440nm]','Lwn_IOP[443nm]','Lwn_IOP[490nm]',
                                            'Lwn_IOP[500nm]','Lwn_IOP[510nm]','Lwn_IOP[531nm]','Lwn_IOP[532nm]',
                                            'Lwn_IOP[551nm]','Lwn_IOP[555nm]','Lwn_IOP[560nm]','Lwn_IOP[620nm]',
                                            'Lwn_IOP[667nm]','Lwn_IOP[675nm]','Lwn_IOP[681nm]','Lwn_IOP[709nm]',
                                            'Lwn_IOP[779nm]','Lwn_IOP[865nm]','Lwn_IOP[870nm]'])
                filter_rad.insert(1, 'AOD', float(aod_value))
                filter_rad.insert(2, 'WV(cm)', float(water_vapor))
                filter_rad.insert(3, 'OZ', float(ozone_value))
                filter_rad.insert(4, 'ALT', float(altitude))
                filter_rad.insert(5, 'CHLA', float(chl))
                list_data.append(filter_rad)
            out = pd.concat(list_data).reset_index()
            out.to_csv(dirout + '/' + file)
        except:
            pass

    return None


def rrs(in_path: str, dest: str) -> None:
    """
    Calculates the Remote Sensing Reflectance and obtain the median spectrum per date.
    """

    dirout = dest + '/tempdir' + '/rrs_median'
    dirout_plot = dest + '/2-plots'

    os.makedirs(dirout, exist_ok=True)
    os.makedirs(dirout_plot, exist_ok=True)

    for file in os.listdir(in_path):
        try:
            dirout_plot_plot = dirout_plot + '/' + file[:-4]
            os.makedirs(dirout_plot_plot, exist_ok=True)

            data = pd.read_csv(in_path + '/' + file, delimiter=',', skiprows=0, engine='python')

            # Solar spectrum from Thuillier (2003) resampled by 11-nm:
            sun_spectrum = {'340': 97.8851, '380': 109.21014500000001, '400': 154.687015, '412': 171.03303499999998,
                            '440': 183.80075000000002,
                            '443': 188.92312, '490': 192.861705, '500': 193.74013, '510': 192.77944499999998,
                            '531': 185.90661500000002,
                            '532': 187.08975999999998, '551': 187.02424500000004, '555': 183.89774, '560': 180.06239,
                            '620': 164.95923499999998,
                            '667': 152.38098499999995, '675': 149.17576999999997, '681': 147.04998500000002,
                            '709': 140.39591000000001, '779': 116.95122,
                            '865': 95.82171000000001, '870': 94.28599499999999, '1020': 69.32279500000001}

            bands_id = ['340', '380', '400', '412', '440', '443', '490', '500', '510', '531',
                        '532', '551', '555', '560', '620', '667', '675', '681', '709', '779', '865', '870']

            # Remote Sensing Reflectance - Rrs (sr^-1):
            rrs = data
            for i in range(0, len(data)):
                arr = np.array(data.loc[i][8:])
                e_sun = np.array([sun_spectrum[i] for i in sun_spectrum])[:-1]
                valid_mask = (arr != -999) & (arr != -9.999)
                rrs_v = np.where(valid_mask, arr / e_sun, -9999)
                rrs.iloc[i, 8:] = rrs_v

            # Returns the median spectrum per measurement:
            dt_total = []
            unique_levels = rrs['Date(dd-mm-yyyy)'].unique()
            for date in unique_levels:
                filter_ = rrs.loc[rrs['Date(dd-mm-yyyy)'] == date].iloc[:, 2:]
                filter_.columns = ['Date', 'AOD', 'WV(cm)', 'OZ', 'ALT', 'CHLA'] + bands_id
                # Removes any spectrum with negative values:
                columns_nan = filter_.columns[(filter_ == -9999).all()]
                df = filter_.drop(columns=columns_nan)
                columns_to_check = df.iloc[:, 6:]
                valid_rows_mask = ~(columns_to_check < 0).any(axis=1)
                df_cleaned = df[valid_rows_mask]

                # Calculates the euclidian distance and recovers the minimum distance between the median reference spectrum and target spectra.
                # Here, we will select a real spectrum closest to the median reference spectrum:
                def euclidianDistance(target_spectrum, reference_spectrum):
                    target_spectrum = np.array(target_spectrum, dtype=float)
                    reference_spectrum = np.array(reference_spectrum, dtype=float)
                    squared_differences = (target_spectrum - reference_spectrum) ** 2
                    return np.sqrt(np.sum(squared_differences))

                if df_cleaned.empty == False:
                    rrs_filter = df_cleaned.iloc[:, 6:]
                    median_reference = pd.DataFrame(rrs_filter.median())
                    spectra = rrs_filter.transpose()
                    start_value = 100  # Default
                    index = []
                    for tag in spectra:
                        euclidian_distance_value = euclidianDistance(spectra[tag], median_reference.transpose())
                        if euclidian_distance_value < start_value:
                            start_value = euclidian_distance_value
                            index.append(tag)
                    median_spect = filter_.loc[filter_.index == index[-1]]
                    dt_total.append(median_spect)
                    arr_median = median_spect.iloc[:, 6:].values
                    arr_median = arr_median[arr_median != -9999]
                    # Plots:
                    rrs_total = df_cleaned.iloc[:, 6:]
                    wavebands = rrs_total.columns
                    rrs_total = rrs_total.transpose()
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
                    sns.lineplot(x=wavebands, y=arr_median, legend=True, linewidth=0.7, color='black', ax=ax, label='Median Spectrum')
                    for spectrum in rrs_total:
                        arr_spectrum = rrs_total[spectrum].values
                        ax.set_title(date, fontsize=8)
                        sns.lineplot(x=wavebands, y=arr_spectrum, legend=True, linewidth=0.5, linestyle='-',
                                     color='gray', alpha=0.5, ax=ax, label='All Spectra')
                        ax.set_ylabel('rrs $(sr^{-1})$', fontsize=6)
                        ax.set_xlabel('wavelength (nm)', fontsize=6)
                        ax.tick_params(axis='both', which='major', labelsize=6)
                    plt.legend()
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), frameon=False, fontsize='x-small')
                    plt.savefig(dirout_plot_plot + '/' + date[0:2] + '-' + date[3:5] + '-' + date[6:10] + '.jpeg',
                                dpi=300, bbox_inches="tight")
                    plt.close()
            dt = pd.concat(dt_total)

            if dt.empty == False:
                dt['Date'] = [i[3:5] + '-' + i[0:2] + '-' + i[6:10] for i in dt['Date']]
                dt.to_csv(dirout + '/' + file, sep=',', index=False)
        except:
            pass

    return None


def owts(in_path: str, dest: str) -> None:
    """
    Classifies the OWTs.
    """

    dirout = dest + '/3-rrs_median'
    os.makedirs(dirout, exist_ok=True)

    owts = pd.read_csv(r'source/OWT_Wei2022.csv', sep=',')

    for file in os.listdir(in_path):
        if '._' not in file:
            rrs = pd.read_csv(in_path + '/' + file)
            bands_id = ['400', '412', '440', '443', '490', '500', '510', '531', '532', '551', '555', '560', '620',
                        '667', '675', '681']
            list_out = []
            for spectrum in range(0, len(rrs)):
                filter_ = rrs.loc[rrs.index == spectrum]
                filter_wave = filter_.filter(bands_id)
                columns_nan = filter_wave.columns[(filter_wave == -9999).all()]
                df = filter_wave.drop(columns=columns_nan)
                arr = df.values.flatten()
                nr_target = arr / np.sqrt(np.sum(arr ** 2))
                owts_filter = owts.filter(df.columns.to_list())

                # Spectra Score - Filters the spectra:
                def SAM(t_spectrum, r_spectrum, nb):
                    """
                    It calculates the Spectral Angle Mapper metric.
                    """
                    # List data:
                    l_multiply = []
                    l_power_t = []
                    l_power_r = []
                    # Calculates the products:
                    for num in range(0, nb):
                        multiply_ = np.multiply(t_spectrum[num], float(r_spectrum[num]))
                        power_t_ = np.power(t_spectrum[num], 2)
                        power_r_ = np.power(float(r_spectrum[num]), 2)
                        l_multiply.append(multiply_)
                        l_power_t.append(power_t_)
                        l_power_r.append(power_r_)
                    # Sums the data:
                    sum_m_ = np.sum(l_multiply, axis=0)
                    sum_t_ = np.sum(l_power_t, axis=0)
                    sum_r_ = np.sum(l_power_r, axis=0)
                    # Calculates the SAM algorithm:
                    factor_1 = np.sqrt(np.multiply(sum_t_, sum_r_))
                    factor_2 = np.divide(sum_m_, factor_1)
                    sam_ = np.arccos(factor_2)
                    return (sam_)

                list_sam = []
                list_wt = []
                for wt in range(0, len(owts)):
                    arr_ref = np.array(owts_filter.loc[wt][:])
                    nr_reference = arr_ref / np.sqrt(np.sum(arr_ref ** 2))
                    list_sam.append(SAM(nr_target, nr_reference, len(nr_reference)))
                    list_wt.append(wt + 1)
                min_index = list_sam.index(np.min(list_sam))
                filter_.insert(6, 'SS', float(np.min(list_sam)))
                filter_.insert(7, 'OWT', str(list_wt[min_index]))
                list_out.append(filter_)
        out = pd.concat(list_out)
        out.to_csv(dirout + '/' + file)

    return None


def shiftband_corr(in_path: str, sensor_type: str, dest: str) -> None:
    """
    Corrects the spectral bands from AERONET-OC to orbital sensors.
    """

    dirout = dest + '/4-' + str(sensor_type)
    os.makedirs(dirout, exist_ok=True)

    # Verification the sensor type:
    aero_all = [340, 380, 400, 412, 440, 443, 490, 500, 510, 531, 532, 551, 555, 560, 620, 667, 675, 681, 709, 779, 865, 870]
    if sensor_type == 'S2A_MSI':
        path = r"source/S2A_MSI_coeff.csv"
        aero_bands = [443, 490, 560, 667, 709, 779, 865]
        sensor_bands = [444, 497, 560, 665, 704, 782, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'S2B_MSI':
        path = r"source/S2B_MSI_coeff.csv"
        aero_bands = [443, 490, 560, 667, 709, 779, 865]
        sensor_bands = [444, 497, 560, 665, 704, 782, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'S3A_OLCI':
        path = r"source/S3A_OLCI_coeff.csv"
        aero_bands = [400, 412, 443, 490, 510, 560, 620, 667, 675, 681, 709, 779, 865]
        sensor_bands = [400, 412, 442, 490, 510, 560, 620, 665, 674, 681, 709, 779, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'S3B_OLCI':
        path = r"source/S3B_OLCI_coeff.csv"
        aero_bands = [400, 412, 443, 490, 510, 560, 620, 667, 675, 681, 709, 779, 865]
        sensor_bands = [400, 412, 442, 490, 510, 560, 620, 665, 674, 681, 709, 779, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'L8_OLI':
        path = r"source/L8_OLI_coeff.csv"
        aero_bands = [440, 490, 560, 667, 865]
        sensor_bands = [440, 480, 560, 655, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'L9_OLI2':
        path = r"source/L9_OLI2_coeff.csv"
        aero_bands = [440, 490, 560, 667, 865]
        sensor_bands = [440, 480, 560, 655, 865]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    elif sensor_type == 'PACE_OCI':
        path = r"source/PACE_OCI_coeff.csv"
        aero_bands = [400, 412, 440, 443, 490, 500, 510, 531, 532, 551, 555, 560, 620, 667, 675, 681, 709, 779, 865, 870]
        sensor_bands = [400, 412, 439, 442, 489, 499, 509, 529, 532, 552, 554, 560, 620, 667, 675, 681, 709, 779, 864, 869]
        missing_bands = [str(band) for band in list(set(aero_all) - set(aero_bands))]
    else:
        print('Sensor not found!')

    # Corrects the AERONET-OC bands:
    table_coeff = pd.read_csv(path, sep=',')
    unique_levels = table_coeff['OWT'].unique()
    unique_levels.sort()

    for file in os.listdir(in_path):
        try:
            data = pd.read_csv(in_path + '/' + file)
            list_out = []
            for i in unique_levels:
                coeff_filter_owt = table_coeff.loc[table_coeff['OWT'] == i]
                rrs_filter = data.loc[data['OWT'] == i].copy()  # Use .copy() to avoid SettingWithCopyWarning
                if not coeff_filter_owt.empty:
                    for aerb, senb in zip(aero_bands, sensor_bands):
                        coeff_filter_v = coeff_filter_owt.loc[coeff_filter_owt['band'] == senb]
                        if not coeff_filter_v.empty:
                            beta_1 = float(coeff_filter_v['b1'].iloc[0])
                            beta_2 = float(coeff_filter_v['b2'].iloc[0])
                            rrs_filter[str(aerb)] = rrs_filter[str(aerb)].apply(
                                lambda x: (beta_1 * x) + beta_2 if x != -9999 else -9999)
                        else:
                            rrs_filter[str(aerb)] = -9999  # without coefficient correction
                    rrs_filter.loc[:, missing_bands] = -9999 # missing bands
                    list_out.append(rrs_filter)
            out = pd.concat(list_out).reset_index(drop=True)
            out.to_csv(os.path.join(dirout, file), index=False)
        except:
            pass

    return None


def shapefile(in_path: str, dest: str) -> None:
    """
    Exports the station point as shapefile (.shp).
    """
    # Creating a directory:
    dirout = dest + '/1-shapefile'
    os.makedirs(dirout, exist_ok=True)

    for file in os.listdir(in_path):
        try:
            # Opens the files:
            data = pd.read_csv(dest + '/0-rawdata' + '/' + file[:-4], delimiter=',', skiprows=5, engine='python')

            latitude = data['Site_Latitude(Degrees)'].values[0]
            longitude = data['Site_Longitude(Degrees)'].values[0]

            point = Point(longitude, latitude)

            gdf = gpd.GeoDataFrame([{'geometry': point}], crs="EPSG:4326")
            gdf.to_file(dirout + '/' + file[:-4] + '.shp', driver='ESRI Shapefile')
        except:
            pass

    return None

