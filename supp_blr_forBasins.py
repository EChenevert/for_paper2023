from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from random import seed

from sklearn.preprocessing import StandardScaler

import main
import pandas as pd
import numpy as np
import funcs
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, cross_val_predict, \
    cross_validate, KFold
import seaborn as sns
import matplotlib


# Everything I need for this should be within the file "D:\Etienne\fall2022\agu_data"
## Data from CIMS
data = main.load_data()
bysite = main.average_bysite(data)


## Data from CRMS
perc = pd.read_csv(r"D:\Etienne\fall2022\agu_data\percentflooded.csv",
                   encoding="unicode escape")
perc['Simple site'] = [i[:8] for i in perc['Station_ID']]
perc = perc.groupby('Simple site').median()
wl = pd.read_csv(r"D:\Etienne\fall2022\agu_data\waterlevelrange.csv",
                 encoding="unicode escape")[['Station_ID', 'Tide_Amp (ft)']]
wl['Simple site'] = [i[:8] for i in wl['Station_ID']]
wl = wl.groupby('Simple site').median()

marshElev = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12009_Survey_Marsh_Elevation\12009_Survey_Marsh_Elevation.csv",
                        encoding="unicode escape").groupby('SiteId').median().drop('Unnamed: 4', axis=1)
SEC = pd.read_csv(r"D:\Etienne\fall2022\agu_data\12017_SurfaceElevation_ChangeRate\12017.csv",
                  encoding="unicode escape")
SEC['Simple site'] = [i[:8] for i in SEC['Station_ID']]
SEC = SEC.groupby('Simple site').median().drop('Unnamed: 4', axis=1)

acc = pd.read_csv(r"D:\Etienne\fall2022\agu_data\12172_SEA\Accretion__rate.csv", encoding="unicode_escape")[
    ['Site_ID', 'Acc_rate_fullterm (cm/y)']
].groupby('Site_ID').median()


## Data from Gee and Arc
jrc = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_GEE_JRCCOPY2.csv", encoding="unicode_escape")[
    ['Simple_sit', 'Land_Lost_m2']
].set_index('Simple_sit')

gee = pd.read_csv(r"D:\Etienne\fall2022\agu_data\CRMS_GEE60pfrom2007to2022.csv",
                          encoding="unicode escape")[['Simple_sit', 'NDVI', 'tss_med', 'windspeed']]\
    .groupby('Simple_sit').median().fillna(0)  # filling nans with zeros cuz all nans are in tss because some sites are not near water


# ############# Attempting the SAVI switch #########################
# gee = pd.read_csv(r"D:\Etienne\fall2022\agu_data\CRMS_GEE60perc_wSAVI.csv",
#                           encoding="unicode escape")[['CRMS Site', 'SAVI', 'tss_med', 'Windspeed (m/s)']]\
#     .groupby('CRMS Site').median().fillna(0)  # filling nans with zeros cuz all nans are in tss because some sites are not near water
# ########################################################################

distRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv",
                        encoding="unicode escape")[['Field1', 'distance_to_river_m', 'width_mean']].groupby('Field1').median()
nearWater = pd.read_csv(r"D:\Etienne\fall2022\agu_data\ALLDATA2.csv", encoding="unicode_escape")[
    ['Simple site', 'Distance_to_Water_m', 'Distance_to_Ocean_m']
].set_index('Simple site')
# Add flooding frequency
floodfreq = pd.read_csv(r"D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\floodingsplits\final_floodfreq.csv", encoding="unicode_escape")[[
    'Simple site', 'Flood Freq (Floods/yr)'
]].set_index('Simple site')
# add flood depth when flooded
floodDepth = pd.read_csv(r"D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\flooddepthsplits\final_flooddepths.csv", encoding="unicode_escape")[[
    'Simple site', 'Avg. Flood Depth when Flooded (ft)', '90th Percentile Flood Depth when Flooded (ft)',
    '10th Percentile Flood Depth when Flooded (ft)', 'Std. Deviation Flood Depth when Flooded '
]].set_index('Simple site')

# Concatenate
df = pd.concat([bysite, distRiver, nearWater, gee, jrc, wl, perc, SEC, floodfreq, floodDepth, acc, marshElev],
               axis=1, join='outer')

df.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\supp_basins_minimal_preprocessing.csv")


# Now clean the columns
# First delete columns that are more than 1/2 nans
tdf = df.dropna(thresh=df.shape[0]*0.5, how='all', axis=1)
# tdf = df.dropna(thresh=df.shape[0]*0.3, how='all', axis=1)  # this threshold lets sand, silt, clay terms stay
# Drop uninformative features
udf = tdf.drop([
    'Year (yyyy)', 'Accretion Measurement 1 (mm)', 'Year',
    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    'Accretion Measurement 4 (mm)',
    'Month (mm)', 'Average Accretion (mm)', 'Delta time (days)', 'Wet Volume (cm3)',
    'Delta Time (decimal_years)', 'Wet Soil pH (pH units)', 'Dry Soil pH (pH units)', 'Dry Volume (cm3)',
    'Measurement Depth (ft)', 'Plot Size (m2)', '% Cover Shrub', '% Cover Carpet', 'Direction (Collar Number)',
    'Direction (Compass Degrees)', 'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
    'percent_waterlevel_complete',  # 'calendar_year',
    'Average Height Shrub (cm)', 'Average Height Carpet (cm)'  # I remove these because most values are nan and these vars are unimportant really

], axis=1)


# Address the vertical measurement for mass calculation (wit the potential of switching between my accretion and
# CRMS accretion)
vertical = 'Accretion Rate (mm/yr)'
if vertical == 'Accretion Rate (mm/yr)':
    udf = udf.drop('Acc_rate_fullterm (cm/y)', axis=1)
    # Make sure multiplier of mass acc is in the right units
    # udf['Average_Ac_cm_yr'] = udf['Accretion Rate (mm/yr)'] / 10  # mm to cm conversion
    # Make sure subsidence and RSLR are in correct units
    udf['Shallow Subsidence Rate (mm/yr)'] = udf[vertical] - udf['Surface Elevation Change Rate (cm/y)'] * 10
    udf['Shallow Subsidence Rate (mm/yr)'] = [0 if val < 0 else val for val in udf['Shallow Subsidence Rate (mm/yr)']]
    udf['SEC Rate (mm/yr)'] = udf['Surface Elevation Change Rate (cm/y)'] * 10
    # Now calcualte subsidence and RSLR
    # Make the subsidence and rslr variables: using the
    udf['SLR (mm/yr)'] = 2.0  # from jankowski
    udf['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * udf['Latitude']) - 114.26) * -1
    udf['RSLR (mm/yr)'] = udf['Shallow Subsidence Rate (mm/yr)'] + udf['Deep Subsidence Rate (mm/yr)'] + udf[
        'SLR (mm/yr)']
    udf = udf.drop(['SLR (mm/yr)'],
                   axis=1)  # obviously drop because it is the same everywhere ; only used for calc

elif vertical == 'Acc_rate_fullterm (cm/y)':
    udf = udf.drop('Accretion Rate (mm/yr)', axis=1)
    #  Make sure multiplier of mass acc is in the right units
    # udf['Average_Ac_cm_yr'] = udf[vertical]
    # Make sure subsidence and RSLR are in correct units
    udf['Shallow Subsidence Rate (mm/yr)'] = (udf[vertical] - udf['Surface Elevation Change Rate (cm/y)'])*10
    udf['SEC Rate (cm/yr)'] = udf['Surface Elevation Change Rate (cm/y)']
    # Now calcualte subsidence and RSLR
    # Make the subsidence and rslr variables: using the
    udf['SLR (mm/yr)'] = 2.0  # from jankowski
    udf['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * udf['Latitude']) - 114.26) * -1
    udf['RSLR (mm/yr)'] = udf['Shallow Subsidence Rate (mm/yr)'] + udf['Deep Subsidence Rate (mm/yr)'] + udf[
        'SLR (mm/yr)']*0.1
    udf = udf.drop(['SLR (mm/yr)'],
                   axis=1)  # obviously drop because it is the same everywhere ; only used for calc
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

####### Define outcome as vertical component
outcome = vertical

udf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\supp_basins_dataset_noOutlierRm.csv")
# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
udf['distance_to_ocean_km'] = udf['Distance_to_Ocean_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', 'Distance_to_Ocean_m',
                'Soil Specific Conductance (uS/cm)',
                'Soil Porewater Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'TSS (mg/L)'})

# Delete the swamp sites and unammed basin
udf.drop(udf.index[udf['Community'] == 'Swamp'], inplace=True)
# udf.drop(udf.index[udf['Basins'] == 'Unammed_basin'], inplace=True)

### want to keep Basins NOW!!! #### for supplementary
# udf = udf.drop('Basins', axis=1)
# ----
udf = udf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
    'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    # taking out water level features because they are not super informative
    # Putting Human in the loop
    'Staff Gauge (ft)', 'Soil Salinity (ppt)',
    'river_width_mean_km',   # 'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution

    # Delete the dominant herb cuz of rendundancy with dominant veg
    'Average Height Herb (cm)',
    # Test delete
    # '10th Percentile Flood Depth when Flooded (ft)', '90th Percentile Flood Depth when Flooded (ft)',
    # other weird ones
    'Soil Porewater Temperature (°C)',
    'Average_Marsh_Elevation (ft. NAVD88)',
     'Organic Density (g/cm3)',  # 'Bulk Density (g/cm3)',
    'Soil Moisture Content (%)',  # 'Organic Matter (%)',  # do not use organic matter because it has a negative relationship, hard for me to interpret --> i think just picks up the bulk density relationship. Or relationship that sites with higher organic matter content tend to have less accretion
    'land_lost_km2'
], axis=1)
# conduct outlier removal which drops all nans
# rdf = funcs.informed_outlierRm(udf.drop(['Community', 'Latitude', 'Longitude', 'Bulk Density (g/cm3)',
#                                          'Organic Matter (%)'], axis=1), thres=3, num=1)
# rdf = funcs.informed_outlierRm(udf.drop(['Community', 'Latitude', 'Longitude', 'Bulk Density (g/cm3)',
#                                          'Organic Matter (%)'], axis=1), thres=2, num=2)
# rdf = funcs.informed_outlierRm(udf.drop(['Community', 'Latitude', 'Longitude',  # 'Bulk Density (g/cm3)', 'Organic Matter (%)'
#                                          ], axis=1), thres=10, num=1)
rdf = funcs.max_interquartile_outlierrm(udf.drop(['Community',
                                                  'Basins',
                                                  'Latitude', 'Longitude', 'Bulk Density (g/cm3)',
                                                  'Organic Matter (%)'], axis=1).dropna(), outcome)
# rdf = funcs.outlierrm_outcome(udf.drop(['Community', 'Latitude', 'Longitude',  # 'Bulk Density (g/cm3)', 'Organic Matter (%)'
#                                          ], axis=1), thres=2, target='Shallow Subsidence Rate (mm/yr)')
# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
# rdf['log_river_width_mean_km'] = [np.log(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
rdf['log_distance_to_river_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
rdf['log_distance_to_ocean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_ocean_km']]
# rdf['Average Height Dominant (mm)'] = rdf['Average Height Dominant (cm)'] * 10
# rdf['Average Height Herb (mm)'] = rdf['Average Height Herb (cm)'] * 10
# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km', 'distance_to_ocean_km'], axis=1)  # 'distance_to_ocean_km'

# Rename some variables for better text wrapping
rdf = rdf.rename(columns={
    'Tide_Amp (ft)': 'Tide Amp (ft)',
    'avg_percentflooded (%)': 'Avg. Time Flooded (%)',
    'windspeed': 'Windspeed (m/s)',

    'log_distance_to_ocean_km': 'Log Distance to Ocean (km)',
    'log_distance_to_water_km': 'Log Distance to Water (km)',
    'log_distance_to_river_km': 'Log Distance to River (km)',
    # My flood depth vars
    '90th Percentile Flood Depth when Flooded (ft)': '90th Percentile Flood Depth (ft)',
    '10th Percentile Flood Depth when Flooded (ft)': '10th Percentile Flood Depth (ft)',
    'Avg. Flood Depth when Flooded (ft)': 'Avg. Flood Depth (ft)',
    'Std. Deviation Flood Depth when Flooded ': 'Std. Deviation Flood Depth (ft)'
})

gdf = pd.concat([rdf, udf[['Community',
                           'Basins',
                           'Latitude', 'Longitude', 'Organic Matter (%)', 'Bulk Density (g/cm3)']]],
                axis=1, join='inner')
# Transform all units to SI units
gdf['Tidal Amplitude (cm)'] = gdf['Tide Amp (ft)'] * 30.48
gdf['90th Percentile Flood Depth (cm)'] = gdf['90th Percentile Flood Depth (ft)'] * 30.48
gdf['10th Percentile Flood Depth (cm)'] = gdf['10th Percentile Flood Depth (ft)'] * 30.48
gdf['Avg. Flood Depth (cm)'] = gdf['Avg. Flood Depth (ft)'] * 30.48
gdf['Std. Deviation Flood Depth (cm)'] = gdf['Std. Deviation Flood Depth (ft)'] * 30.48

# Delete the old non SI unit variables
# gdf = gdf.drop(['Std. Deviation Flood Depth (ft)', 'Avg. Flood Depth (ft)', 'Tide Amp (ft)'], axis=1)
gdf = gdf.drop(['Std. Deviation Flood Depth (ft)', 'Avg. Flood Depth (ft)', '10th Percentile Flood Depth (ft)',
                '90th Percentile Flood Depth (ft)', 'Tide Amp (ft)'], axis=1)

# Export gdf to file specifically for AGU data and results
gdf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\supp_basins_dataset.csv")
# gdf = gdf.drop('distance_to_ocean_km', axis=1)  # why?
# split into marsh datasets

ponchartraindf = gdf[gdf['Basins'] == 'Ponchartrain']
# unnameddf = gdf[gdf['Community'] == 'Unnamed_basin']
brentondf = gdf[gdf['Basins'] == 'Brenton Sound']
#mrddf = gdf[gdf['Basins'] == 'MRD']  # MRD excluded because sample size is too small after outlier removal
baratariadf = gdf[gdf['Basins'] == 'Barataria']
terrebonnedf = gdf[gdf['Basins'] == 'Terrebonne']
atchafalayadf = gdf[gdf['Basins'] == 'Atchafalaya']
tech_vermdf = gdf[gdf['Basins'] == 'Teche_Vermillion']
mermdf = gdf[gdf['Basins'] == 'Mermentau']
calc_sabdf = gdf[gdf['Basins'] == 'Calcasieu_Sabine']

# We do not include sites with no assocaited basin (this have limited info anyway)

directly_supplied = gdf[(gdf['Basins'] == 'MRD') | (gdf['Basins'] == 'Atchafalaya')]
indirectly_supplied = gdf[(gdf['Basins'] != 'MRD') | (gdf['Basins'] != 'Atchafalaya')
                          | (gdf['Basins'] != np.nan) | (gdf['Basins'] != 'Unnamed_basin')]

marshdic = {'All': gdf, 'Ponchartrain': ponchartraindf, 'BrentonSound': brentondf,
            # 'MRD': mrddf,  # Excluded because not enough data for cross-validation assessment and linaer regression significance
            'Barataria': baratariadf,
            'Terrebonne': terrebonnedf,
            'Atchafalaya': atchafalayadf,
            'TecheVermillion': tech_vermdf,
            'Mermentau': mermdf,
            # 'CalcasieuSabine': calc_sabdf, # Excluded because no features were deemed below the set significance threshold in the backward elimination process
            'Indirectly_supplied': indirectly_supplied,
            'directly_supplied': directly_supplied}


hold_marsh_weights = {}
hold_unscaled_weights = {}
hold_intercept = {}
hold_marsh_regularizors = {}
hold_marsh_weight_certainty = {}
hold_prediction_certainty = {}

for key in marshdic:
    print(key)
    mdf = marshdic[key]  # .drop('Community', axis=1)
    # It is preshuffled so i do not think ordering will be a problem
    # t = np.log10(mdf[outcome].reset_index().drop('index', axis=1))
    t = mdf[outcome].reset_index().drop('index', axis=1)
    phi = mdf.drop([outcome, 'Community',
                    'Basins',
                    'Latitude', 'Longitude',  'Organic Matter (%)', 'Bulk Density (g/cm3)',
                    ],
                   axis=1).reset_index().drop('index', axis=1)
    # Scale: because I want feature importances
    scalar_Xmarsh = StandardScaler()
    predictors_scaled = pd.DataFrame(scalar_Xmarsh.fit_transform(phi), columns=phi.columns.values)
    # # NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
    # br = linear_model.BayesianRidge(fit_intercept=True)
    #
    # feature_selector = ExhaustiveFeatureSelector(br,
    #                                                  min_features=1,
    #                                                  max_features=len(phi.columns.values),
    #                                                  # I should only use 5 features (15 takes waaaaay too long)
    #                                                  scoring='neg_mean_absolute_error',
    #                                                  # print_progress=True,
    #                                                  cv=3)  # 3 fold cross-validation
    #
    # efsmlr = feature_selector.fit(predictors_scaled, t.values.ravel())
    #
    # print('Best CV r2 score: %.2f' % efsmlr.best_score_)
    # print('Best subset (indices):', efsmlr.best_idx_)
    # print('Best subset (corresponding names):', efsmlr.best_feature_names_)
    #
    # bestfeaturesM = list(efsmlr.best_feature_names_)

    bestfeaturesM = funcs.backward_elimination(predictors_scaled, t, num_feats=20, significance_level=0.05)

    # bestfeaturesM = funcs.backward_elimination(predictors_scaled, t.values.ravel(), num_feats=100,
    #                                            significance_level=0.01)

    # Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val
    #### NOTE: I should do separate tests to determine which split of the data is optimal ######
    # first split data set into test train
    from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

    X, y = predictors_scaled[bestfeaturesM], t

    baymod = linear_model.BayesianRidge(fit_intercept=True)

    results_dict = funcs.cv_results_and_plot_PATH(baymod, bestfeaturesM, phi, X, y,
                                                  {'cmap': 'YlOrRd', 'line': "r--"}, str(key),
                                                  "D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\")

    hold_marsh_weights[key] = results_dict["Scaled Weights"]
    hold_unscaled_weights[key] = results_dict["Unscaled Weights"]
    hold_marsh_regularizors[key] = results_dict["Scaled regularizors"]
    hold_marsh_weight_certainty[key] = results_dict["# Well Determined Weights"]
    hold_prediction_certainty[key] = results_dict["Standard Deviations of Predictions"]
    hold_intercept[key] = results_dict["Unscaled Intercepts"]

# Make a colormap so all each weight will have a specific color
colormap = {
'Soil Porewater Salinity (ppt)': '#DD8A8A',
'Average Height Dominant (cm)': '#137111',
'NDVI': '#0AFF06',
'TSS (mg/L)': '#8E6C02',
'Windspeed (m/s)': '#70ECE3',
'Tidal Amplitude (cm)': '#434F93',
'Avg. Flood Depth (cm)': '#087AFA',
'SAVI':  '#087AFD',
# '90th Percentile of Waterlevel to Marsh (ft)': '#D001A1',
'90th Percentile Flood Depth (cm)': '#D000E1',
# '10th Percentile of Waterlevel to Marsh (ft)': '#73ABAE',
'10th Percentile Flood Depth (cm)': '#73ACAE',
# 'Std. Deviation of Flooding (ft)': '#DE5100',
'Std. Deviation Flood Depth (cm)': '#DE5100',
'Avg. Time Flooded (%)': '#970CBD',
'Flood Freq (Floods/yr)': '#EB0000',
'Log Distance to Water (km)': '#442929',
'Log Distance to River (km)': '#045F38',
'Log Distance to Ocean (km)': '#045F27'
}

for key in hold_marsh_weights:
    d = pd.DataFrame(hold_marsh_weights[key].mean().reset_index()).rename(columns={0: 'Means'})
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_ylabel("Relative Feature Importance")
    # my_cmap = plt.get_cmap("cool")
    # ax.bar(list(d['index']), list(d['Means']), color='Blue')
    ax.set_title(str(key) + " CRMS Stations", fontsize=21)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # sns.barplot(data=hold_marsh_weights[key], palette="Blues")
    palette_ls = []
    for weight in d['index']:
        palette_ls.append(colormap[weight])
    sns.barplot(list(d['index']), list(d['Means']), palette=palette_ls)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\" + str(key) +
               "_supp_scaledX_nolog_boxplot_human.eps", format='eps',
               dpi=300,
               bbox_inches='tight')
    fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\" + str(key) +
               "_supp_scaledX_nolog_boxplot_human.png", format='png',
               dpi=300,
               bbox_inches='tight')
    plt.show()

# Plot the distribution of weight parameters for the marsh runs
for key in hold_unscaled_weights:
    print("Unscaled Weights for " + str(key))
    print(hold_unscaled_weights[key].mean())
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_ylabel("Rescaled Weight Coefficients")
    # matplotlib.rcParams['pdf.fonttype'] = 42
    ax.set_title(str(key) + " CRMS Stations")
    ax.axhline(0, ls='--')
    # if key != 'Saline':
    #     ax.axhline(0, ls='--')
    palette_ls = []
    for weight in hold_unscaled_weights[key].keys():
        palette_ls.append(colormap[weight])
    boxplot = sns.boxplot(data=hold_unscaled_weights[key], notch=True, showfliers=False, palette=palette_ls, width=0.4)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\" + str(
       key) + "_supp_unscaledWeights_nolog_boxplot_human.eps", format='eps',
               dpi=300,
               bbox_inches='tight')
    plt.show()


# Plot the distribution of the eff_reg parameter for each run
eff_reg_df = pd.DataFrame(hold_marsh_regularizors)
sns.set_theme(style='white', font_scale=1)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Learned Effective Regularization Parameters')
sns.boxplot(data=eff_reg_df, notch=True, showfliers=False, palette="YlOrBr")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\supp_regularization_scaledX_nolog_boxplot_human.eps",
           format='eps',
           dpi=300,
           bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of parameters for each run
certainty_df = pd.DataFrame(hold_marsh_weight_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Calculated Number of Well Determined Parameters')
sns.boxplot(data=certainty_df, notch=True, showfliers=False, palette="Blues")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\supp_certainty_scaledX_nolog_boxplot_human.eps",
           format='eps',
           dpi=300,
           bbox_inches='tight')
plt.show()



# Plot the distribution calculated intercepts
intercept_df = pd.DataFrame(hold_intercept)
sns.set_theme(style='white', rc={'figure.dpi': 147}, font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Intercepts [Unscaled]:')
ax.axhline(0, ls='--')
sns.boxplot(data=intercept_df, notch=True, showfliers=False, palette="coolwarm")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\supp_intercepts_nolog_boxplot_human.eps", dpi=300,
           format='eps',
           bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of predictions for each run
pred_certainty_df = pd.DataFrame(hold_prediction_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Bayesian Uncertainty in Predictions')
sns.boxplot(data=pred_certainty_df, notch=True, showfliers=False, palette="Reds")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\PAPER_2023\\additional_supplemental\\blr_per_basin\\pred_certainty_scaledX_nolog_boxplot_human.eps",
           dpi=300, format='eps',
           bbox_inches='tight')
plt.show()

# Following https://christophm.github.io/interpretable-ml-book/limo.html for individual feature importances
# Want to show points for the 10th, 25th, 50th, 75th, 90th poins of outcome and their feature effects
