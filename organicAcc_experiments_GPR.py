from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
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
                 encoding="unicode escape")[['Station_ID', 'Tide_Amp (ft)', '10%thLower_flooding (ft)',
                                             '90%thUpper_flooding (ft)', 'avg_flooding (ft)']]
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
    ['Simple site', 'Distance_to_Water_m']  # 'Distance_to_Ocean_m'
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
df = pd.concat([bysite, distRiver, nearWater, gee, jrc, marshElev, wl, perc, SEC, acc, floodfreq, floodDepth],
               axis=1, join='outer')


# Make mineral Density var from organic density
# Convert Average accretion to cm
df['Average_Ac_cm_yr'] = df['Accretion Rate (mm/yr)'] / 10  # mm to cm conversion
# d['Mineral_De'] = d['Bulk_Densi'] - d['Organic_De']
# d['Bulk Accumulation (g/cm3)'] = d['Bulk_Densi'] * d['Average_Ac_cm'] * 10000  # Equation from Nyman et al 2006
# d['Organic Accumulation (g/cm3)'] = d['Bulk_Densi'] * d['Average_Ac_cm'] * 10000  # Equation from Nyman et al 2006
A = 10000  # This is the area of the study, in our case it is per site, so lets say the area is 1 m2 in cm
df['Total Mass Accumulation (g/yr)'] = (df['Bulk Density (g/cm3)'] * df['Average_Ac_cm_yr']) * A  # g/cm3 * cm/yr * cm2 = g/yr
df['Organic Mass Accumulation (g/yr)'] = (df['Bulk Density (g/cm3)'] * df['Average_Ac_cm_yr'] * (df['Organic Matter (%)']/100)) * A
df['Mineral Mass Accumulation (g/yr)'] = df['Total Mass Accumulation (g/yr)'] - df['Organic Mass Accumulation (g/yr)']
df['Organic Mass Accumulation Fraction'] = df['Organic Mass Accumulation (g/yr)']/df['Total Mass Accumulation (g/yr)']




# Now clean the columns
# First delete columns that are more than 1/2 nans
tdf = df.dropna(thresh=df.shape[0]*0.5, how='all', axis=1)
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





# Address the vertical measurement for mass calculation (multiple potential outcome problem)
####### Define outcome as vertical component
outcome = "Organic Mass Accumulation (g/yr)"
# Drop other potential outcome varaibles that are well related
udf = udf.drop(["Mineral Mass Accumulation (g/yr)", "Total Mass Accumulation (g/yr)",
                "Accretion Rate (mm/yr)", 'Average_Ac_cm_yr', 'Surface Elevation Change Rate (cm/y)',
                'Organic Mass Accumulation Fraction', 'Acc_rate_fullterm (cm/y)'], axis=1)


# udf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\orgAcc_dataset_noOutlierRm.csv")
# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
# udf['distance_to_ocean_km'] = udf['Distance_to_Ocean_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', #  'Distance_to_Ocean_m',
                'Soil Specific Conductance (uS/cm)',
                'Soil Porewater Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'TSS (mg/l)'})

# Delete the swamp sites and unammed basin
udf.drop(udf.index[udf['Community'] == 'Swamp'], inplace=True)
# udf.drop(udf.index[udf['Basins'] == 'Unammed_basin'], inplace=True)
udf = udf.drop('Basins', axis=1)
# ----
udf = udf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    #### No Longer need wit the by-part (mineral v organic) accumulation rates ###

#     'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
#     'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion

    # taking out water level features because they are not super informative
    # Putting Human in the loop
    # '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
    # 'std_deviation_water_level(ft NAVD88)',
    'Staff Gauge (ft)', 'Soil Salinity (ppt)',


    'river_width_mean_km',   # 'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution

    # Delete the dominant herb cuz of rendundancy with dominant veg
    'Average Height Herb (cm)',
    # 'tss med mg/l',  # cuz idk if i trust calc..... eh
    # # Taking these flood depth variables out becuase I compute them myself better!!!!
    # 'std_deviation_avg_flooding (ft)',  # cuz idk how it differs from tide amp, is diff correlated as well from SHAP
    'avg_flooding (ft)',  # remove because I now calcuate flooding depth when flooded
    '10%thLower_flooding (ft)',  # same reason as above AND i compute myself
    '90%thUpper_flooding (ft)',
    # other weird ones
    'Soil Porewater Temperature (°C)',
    'Average_Marsh_Elevation (ft. NAVD88)',
     'Organic Density (g/cm3)',  # 'Bulk Density (g/cm3)',
    'Soil Moisture Content (%)',  # 'Organic Matter (%)',  # do not use organic matter because it has a negative relationship, hard for me to interpret --> i think just picks up the bulk density relationship. Or relationship that sites with higher organic matter content tend to have less accretion
    'land_lost_km2'
], axis=1)
# conduct outlier removal which drops all nans
rdf = funcs.max_interquartile_outlierrm(udf.drop(['Community', 'Latitude', 'Longitude', 'Bulk Density (g/cm3)',
                                         'Organic Matter (%)'], axis=1).dropna(), target=outcome)
# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
# rdf['log_river_width_mean_km'] = [np.log(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
rdf['log_distance_to_river_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
# rdf['log_distance_to_ocean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_ocean_km']]
# rdf['Average Height Dominant (mm)'] = rdf['Average Height Dominant (cm)'] * 10
# rdf['Average Height Herb (mm)'] = rdf['Average Height Herb (cm)'] * 10
# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km'], axis=1)  # 'distance_to_ocean_km'
# Now it is feature selection time
# drop any variables related to the outcome
# rdf = rdf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
#     'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
#     'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
#     # taking out water level features because they are not super informative
#     # Putting Human in the loop
#     # '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
#     # 'std_deviation_water_level(ft NAVD88)',
#     'Staff Gauge (ft)', 'Soil Salinity (ppt)',
#     'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution
#     # Delete the dominant herb cuz of rendundancy with dominant veg
#     'Average Height Herb (cm)',
#     # 'tss med mg/l',  # cuz idk if i trust calc..... eh
#     # # Taking these flood depth variables out becuase I compute them myself better!!!!
#     # 'std_deviation_avg_flooding (ft)',  # cuz idk how it differs from tide amp, is diff correlated as well from SHAP
#     'avg_flooding (ft)',  # remove because I now calcuate flooding depth when flooded
#     '10%thLower_flooding (ft)',  # same reason as above AND i compute myself
#     '90%thUpper_flooding (ft)',
#     # other weird ones
#     'Soil Porewater Temperature (°C)',
#     'Average_Marsh_Elevation (ft. NAVD88)',
#     'Bulk Density (g/cm3)',  'Organic Density (g/cm3)',
#     'Soil Moisture Content (%)',  'Organic Matter (%)',  # do not use organic matter because it has a negative relationship, hard for me to interpret --> i think just picks up the bulk density relationship. Or relationship that sites with higher organic matter content tend to have less accretion
#     'land_lost_km2'
# ], axis=1)

# Rename some variables for better text wrapping
rdf = rdf.rename(columns={
    'Tide_Amp (ft)': 'Tide Amp (ft)',
    'avg_percentflooded (%)': 'Avg. Time Flooded (%)',
    'windspeed': 'Windspeed (m/s)',
    # 'log_distance_to_ocean_km': 'log distance to ocean km',
    # 'Average_Marsh_Elevation (ft. NAVD88)': 'Average Marsh Elevation (ft. NAVD88)',
    'log_distance_to_water_km': 'Log Distance to Water (km)',
    'log_distance_to_river_km': 'Log Distance to River (km)',
    '10%thLower_flooding (ft)': '10th Percentile of Waterlevel to Marsh (ft)',
    '90%thUpper_flooding (ft)': '90th Percentile of Waterlevel to Marsh (ft)',
    'avg_flooding (ft)': 'Avg. Waterlevel to Marsh (ft)',
    'std_deviation_avg_flooding (ft)': 'Std. Deviation of Flooding (ft)',
    # My flood depth vars
    '90th Percentile Flood Depth when Flooded (ft)': '90th Percentile Flood Depth (ft)',
    '10th Percentile Flood Depth when Flooded (ft)': '10th Percentile Flood Depth (ft)',
    'Avg. Flood Depth when Flooded (ft)': 'Avg. Flood Depth (ft)',
    'Std. Deviation Flood Depth when Flooded ': 'Std. Deviation Flood Depth (ft)'
})

gdf = pd.concat([rdf, udf[['Community', 'Latitude', 'Longitude', 'Organic Matter (%)', 'Bulk Density (g/cm3)']]],
                axis=1, join='inner')
# Transform all units to SI units
gdf['Tidal Amplitude (cm)'] = gdf['Tide Amp (ft)'] * 30.48
gdf['90th Percentile Flood Depth (cm)'] = gdf['90th Percentile Flood Depth (ft)'] * 30.48
gdf['10th Percentile Flood Depth (cm)'] = gdf['10th Percentile Flood Depth (ft)'] * 30.48
gdf['Avg. Flood Depth (cm)'] = gdf['Avg. Flood Depth (ft)'] * 30.48
gdf['Std. Deviation Flood Depth (cm)'] = gdf['Std. Deviation Flood Depth (ft)'] * 30.48

# Delete the old non SI unit variables
gdf = gdf.drop(['Std. Deviation Flood Depth (ft)', 'Avg. Flood Depth (ft)', '10th Percentile Flood Depth (ft)',
                '90th Percentile Flood Depth (ft)', 'Tide Amp (ft)'], axis=1)

# Export gdf to file specifically for AGU data and results
# gdf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\organicAcc\\orgAcc_dataset.csv")

# # ------------------------ Check if i get rid of 0 TSS values ---------------------------------------------
# gdf = gdf[gdf['TSS (mg/l)'] != 0]

### --- Begin the GPR regression --- ###

### We will only compute the GPR for the whole dataset --> found that it was the most efficent
predictors = gdf.drop([outcome, 'Community', 'Longitude', 'Latitude', 'Organic Matter (%)', 'Bulk Density (g/cm3)'], axis=1)
target = gdf[outcome]
# Scale
scalar = StandardScaler()
predictors_scaled = pd.DataFrame(scalar.fit_transform(predictors), columns=predictors.columns.values)
# Set the kernel for GPR

# # Could make the excuse that it is too computationally expensive to do this calculation and therefore backward \
# feature selection is preferable... a lil weird tho... tends to linear relationships

# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, alpha=0.5)
#
# feature_selector = ExhaustiveFeatureSelector(gpr, min_features=1, max_features=len(predictors_scaled.columns.values),
#                                             scoring='neg_mean_absolute_error', cv=3)
#
# efs = feature_selector.fit(predictors_scaled, target.values.ravel())
# print('Best Subset (feature names): ', efs.best_feature_names_)
#
# X = predictors[list(efs.best_feature_names_)]

# # Backward feature elimination
# bestfeatures = funcs.backward_elimination(data=predictors_scaled, target=target, num_feats=20, significance_level=0.05)
# X = predictors_scaled[bestfeatures]

##### I decide to use features that are informed by my split dataset BLR tests
bestfeatures = ['Tidal Amplitude (cm)', 'TSS (mg/l)', 'Soil Porewater Salinity (ppt)', 'NDVI']

X = predictors_scaled[bestfeatures]
#
# kernel = (DotProduct() ** 2) + WhiteKernel()
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, alpha=0.5)
# feature_selector = ExhaustiveFeatureSelector(gpr,
#                                                  min_features=1,
#                                                  max_features=len(predictors_scaled.columns.values),
#                                                  # I should only use 5 features (15 takes waaaaay too long)
#                                                  scoring='neg_mean_absolute_error',
#                                                  # print_progress=True,
#                                                  cv=5)  # 3 fold cross-validation
#
# efsmlr = feature_selector.fit(predictors_scaled, target.values.ravel())
#
# print('Best CV r2 score: %.2f' % efsmlr.best_score_)
# print('Best subset (indices):', efsmlr.best_idx_)
# print('Best subset (corresponding names):', efsmlr.best_feature_names_)
#
# bestfeatures = list(efsmlr.best_feature_names_)
#
# X = predictors_scaled[bestfeatures]

# ### Now for the actual testing.
# rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=123)
# scores = cross_validate(gpr, X, target, cv=rcv)

# Visualize manual cross validation

# Performance Metric Containers: I allow use the median because I want to be more robust to outliers
r2_total_medians = []  # holds the k-fold median r^2 value. Will be length of 100 due to 100 repeats
mae_total_medians = []  # holds the k-fold median Mean Absolute Error (MAE) value. Will be length of 100 due to 100 repeats

predicted = []
y_ls = []

prediction_certainty_ls = []
prediction_list = []

for i in range(100):  # for 100 repeats
    try_cv = KFold(n_splits=5, shuffle=True)

    # errors
    r2_ls = []
    mae_ls = []
    # predictions
    pred_certain = []
    pred_list = []

    for train_index, test_index in try_cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        # Fit the model
        kernel = (DotProduct() ** 2) + WhiteKernel()
        # kernel = DotProduct() + RBF()
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, alpha=0.5)

        gpr.fit(np.asarray(X_train), np.asarray(y_train))
        # predict
        ypred, ystd = gpr.predict(X_test, return_std=True)
        pred_list += list(ypred)
        pred_certain += list(ystd)

        r2 = r2_score(y_test, ypred)
        r2_ls.append(r2)
        mae = mean_absolute_error(y_test, ypred)
        mae_ls.append(mae)

    # Average certainty in predictions
    prediction_certainty_ls.append(np.mean(pred_certain))
    prediction_list.append(pred_list)

    # Average predictions over the Kfold first: scaled
    r2_median = np.median(r2_ls)
    r2_total_medians.append(r2_median)
    mae_median = np.median(mae_ls)
    mae_total_medians.append(mae_median)

    predicted = predicted + list(cross_val_predict(gpr, X, target.values.ravel(), cv=try_cv))
    y_ls += list(target.values.ravel())


# Now calculate the mean of th kfold means for each repeat: scaled accretion
r2_final_median = np.median(r2_total_medians)
mae_final_median = np.median(mae_total_medians)

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(9, 8))
hb = ax.hexbin(x=y_ls,
               y=predicted,
               gridsize=30, edgecolors='grey',
               cmap='YlOrRd', mincnt=1)
ax.set_facecolor('white')
ax.set_xlabel("Measured Accretion Rate (mm/yr)", fontsize=21)
ax.set_ylabel("Estimated Accretion Rate (mm/yr)", fontsize=21)
ax.set_title("All CRMS Stations GPR: Mineral Accumulation Rates", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)
cb = fig.colorbar(hb, ax=ax)
cb.ax.get_yaxis().labelpad = 20
cb.set_label('Density of Predictions', rotation=270, fontsize=21)

ax.plot([target.min(), target.max()], [target.min(), target.max()],
        "k--", lw=3)

ax.annotate("Median r-squared = {:.3f}".format(r2_final_median), xy=(190, 30), xycoords='axes points',
            bbox=dict(boxstyle='round', fc='w'),
            size=15, ha='left', va='top')
ax.annotate("Median MAE = {:.3f}".format(mae_final_median), xy=(190, 60), xycoords='axes points',
            bbox=dict(boxstyle='round', fc='w'),
            size=15, ha='left', va='top')
plt.show()

# fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\cross_validation.eps",
#             format='eps',
#             dpi=300,
#             bbox_inches='tight')


# SHAP analysis
import shap

# Rename X to "Standardized Variables" this way it is clear that the variable distributions are standardized
X = X.rename(columns={'Tidal Amplitude (cm)': 'Tidal Amplitude (*)',
                      'NDVI': 'NDVI (*)',
                      # '90th Percentile Flood Depth (cm)': '90th Percentile Flood Depth (*)',
                      'Avg. Flood Depth (cm)': 'Avg. Flood Depth (*)',
                      'TSS (mg/l)': 'TSS (*)',
                      'Soil Porewater Salinity (ppt)': 'Soil Porewater Salinity (*)'})
# Sampling and shap computation for explanation
gpr.fit(X, target)
X500 = shap.utils.sample(X, 500)
print(type(X500))

explainer = shap.Explainer(gpr.predict, X500)
shap_values = explainer(X)

plt.rcParams['legend.fontsize'] = 21

plt.figure()
# Summary plot
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_size=[10, 5], show=False)
plt.xlabel('mean(|SHAP value|)(average impact on model output magnitude)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\shap_summaryplot.pdf", format="pdf", dpi=300, bbox_inches='tight')

# Bar plot for feature importance
plt.figure()
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_size=[10, 5], show=False, plot_type="bar")
plt.xlabel('mean(|SHAP value|)(average impact on model output magnitude)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\shap_barplot.pdf", format="pdf", dpi=300, bbox_inches='tight')

# Summary heat map
plt.figure()
shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1), show=False)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\summary_heatmap.pdf", format="pdf", dpi=300, bbox_inches='tight')

# Partial and SHAP dependence for Tidal Amplitude
plt.figure()
shap.partial_dependence_plot('Tidal Amplitude (*)', gpr.predict, X500, ice=False, model_expected_value=True,
                             feature_expected_value=True, show=False)
# plt.xlabel("SHAP value for Tidal Amplitude (*)", fontsize=21)
# plt.ylabel("Tidal Amplitude (*)", fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\tidalAmplitude_partial.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.figure()
shap.plots.scatter(shap_values[:, 'Tidal Amplitude (*)'], color=shap_values[:, 'Avg. Flood Depth (*)'],
                   show=False)
plt.xlabel("Tidal Amplitude (*)", fontsize=21)
plt.ylabel("SHAP value for Tidal Amplitude (*)", fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\tidalAmplitude_partialSHAP.pdf", format="pdf", dpi=300, bbox_inches='tight')


# Partial and SHAP dependence for Avg. flood depth
plt.figure()
shap.partial_dependence_plot('Avg. Flood Depth (*)', gpr.predict, X500, ice=False, model_expected_value=True,
                             feature_expected_value=True, show=False)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\90flooddepth_partial.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.figure()
shap.plots.scatter(shap_values[:, 'Avg. Flood Depth (*)'], color=shap_values[:, 'Tidal Amplitude (*)'],
                   show=False)
plt.xlabel('Avg. Flood Depth (*)', fontsize=21)
plt.ylabel('SHAP value for Avg. Flood Depth (*)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\90flodddepth_partialSHAP.pdf", format="pdf", dpi=300, bbox_inches='tight')

# partial and SHAP for NDVI
plt.figure()
shap.partial_dependence_plot('NDVI (*)', gpr.predict, X500, ice=False, model_expected_value=True,
                             feature_expected_value=True, show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\NDVI_partial.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.figure()
shap.plots.scatter(shap_values[:, 'NDVI (*)'], color=shap_values[:, 'Soil Porewater Salinity (*)'],
                   show=False)
plt.xlabel('NDVI (*)', fontsize=21)
plt.ylabel('SHAP value for NDVI (*)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\NDVI_partialSHAP.pdf", format="pdf", dpi=300, bbox_inches='tight')

# ppartial and SHAP for salinity
plt.figure()
shap.partial_dependence_plot('Soil Porewater Salinity (*)', gpr.predict, X500, ice=False, model_expected_value=True,
                             feature_expected_value=True, show=False)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\salinity_partial.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.figure()
shap.plots.scatter(shap_values[:, 'Soil Porewater Salinity (*)'], color=shap_values[:, 'NDVI (*)'],
                   show=False)
plt.xlabel('Soil Porewater Salinity (*)', fontsize=21)
plt.ylabel('SHAP value for Soil Porewater Salinity (*)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\salinity_partialSHAP.pdf", format="pdf", dpi=300, bbox_inches='tight')

# partial and SHAP for TSS
plt.figure()
shap.partial_dependence_plot('TSS (*)', gpr.predict, X500, ice=False, model_expected_value=True,
                             feature_expected_value=True, show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\TSS_partial.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.figure()
shap.plots.scatter(shap_values[:, 'TSS (*)'], color=shap_values[:, 'Soil Porewater Salinity (*)'],
                   show=False)
plt.xlabel('TSS (*)', fontsize=21)
plt.ylabel('SHAP value for TSS (*)', fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\TSS_partialSHAP.pdf", format="pdf", dpi=300, bbox_inches='tight')


# SHAP Dependence Plots for the discussion regarding interactions between tidal amplitude, flood depth, and NDVI
# Tidal + NDVI
plt.figure()
shap.plots.scatter(shap_values[:, 'Tidal Amplitude (*)'], color=shap_values[:, 'NDVI (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\tidal_NDVI_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# Flood Depth + NDVI
plt.figure()
shap.plots.scatter(shap_values[:, 'Avg. Flood Depth (*)'], color=shap_values[:, 'NDVI (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=21)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\flood_NDVI_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# Tidal + Salinity
plt.figure()
shap.plots.scatter(shap_values[:, 'Tidal Amplitude (*)'], color=shap_values[:, 'Soil Porewater Salinity (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\tidal_salinity_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# Flood + Salinity
plt.figure()
shap.plots.scatter(shap_values[:, 'Avg. Flood Depth (*)'], color=shap_values[:, 'Soil Porewater Salinity (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\flood_salinity_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# NDVI + Tide
plt.figure()
shap.plots.scatter(shap_values[:, 'NDVI (*)'], color=shap_values[:, 'Tidal Amplitude (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\NDVI_tide_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# NDVI + flood
plt.figure()
shap.plots.scatter(shap_values[:, 'NDVI (*)'], color=shap_values[:, 'Avg. Flood Depth (*)'],
                   show=False)
# plt.xlabel(fontsize=21)
# plt.ylabel(fontsize=21)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\NDVI_flood_partialSHAP.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')
# I think these interactions make sense if we see the distribution of NDVI values across marsh types (?)
# ---- This would say that saline marshes have the lowest NDVI, thus, these areas are the ones accreting the most...?
# ---- But then we have to explain how...?
# We can also go the route of investigating the NDVI signitures of ...

# Provide example waterfall plot of some of the percentiles of the tidal ampltiude variable
def get_percentile(percentile_decimal, array):
    idx = round(percentile_decimal * (len(array) - 1))
    val = np.sort(array)[idx]
    real_idx = np.where(array == val)
    return real_idx[0]


# When the tidal amplitude has a high influence => 90th, 75th, 50th, 25th, 10th
tide90 = get_percentile(0.9, shap_values.data[:, 0])[0]
tide75 = get_percentile(0.75, shap_values.data[:, 0])[0]
tide50 = get_percentile(0.5, shap_values.data[:, 0])[0]
tide25 = get_percentile(0.25, shap_values.data[:, 0])[0]
tide10 = get_percentile(0.1, shap_values.data[:, 0])[0]

# plt.figure(figsize=(10, 15))
plt.figure()
shap.plots.waterfall(shap_values[tide90], show=False)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\organicAcc\\90thTidalAmp_waterfall.pdf", format="pdf", dpi=300,
#             bbox_inches='tight')


