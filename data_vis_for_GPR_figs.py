import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\\Etienne\\fall2022\\agu_data\\results\\AGU_dataset.csv", encoding='unicode_escape')

df = df.rename(columns={'TSS (mg/l)': 'TSS (mg/L)'})

### Histograms of GPR variables
# Tidal Amplitude
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=df['Tidal Amplitude (cm)'], kde=False,
             hue=df["Community"], palette=['#ADD8E6', '#032180', '#5DC069', '#006E0D'],
             element="bars", legend=True)
# ax.set_title("Distribution of Tidal Amplitude", fontsize=24)
ax.set_xlabel("Tidal Amplitude (cm)", fontsize=21)
ax.set_ylabel("Count", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

f.legend(fontsize=21)
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\hist_tidal.eps",
          dpi=300, format="eps")


# NDVI
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=df['NDVI'], kde=False,
             hue=df["Community"], palette=['#ADD8E6', '#032180', '#5DC069', '#006E0D'],
             element="bars", legend=True)
# ax.set_title("Distribution of NDVI", fontsize=24)
ax.set_xlabel("NDVI", fontsize=21)
ax.set_ylabel("Count", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

f.legend(fontsize=21)
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\hist_ndvi.eps",
          dpi=300, format="eps")


# Soil Porewater Salinity
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=df['Soil Porewater Salinity (ppt)'], kde=False,
             hue=df["Community"], palette=['#ADD8E6', '#032180', '#5DC069', '#006E0D'],
             element="bars", legend=True)
# ax.set_title("Distribution of Soil Porewater Salinity", fontsize=24)
ax.set_xlabel('Soil Porewater Salinity (ppt)', fontsize=21)
ax.set_ylabel("Count", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

f.legend(fontsize=21)
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\hist_salinity.eps",
          dpi=300, format="eps")


# 90th percenitle flood depth
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=df['90th Percentile Flood Depth (cm)'], kde=False,
             hue=df["Community"], palette=['#ADD8E6', '#032180', '#5DC069', '#006E0D'],
             element="bars", legend=True)
# ax.set_title("Distribution of 90th Percentile Flood Depth", fontsize=24)
ax.set_xlabel('90th Percentile Flood Depth (cm)', fontsize=21)
ax.set_ylabel("Count", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

f.legend(fontsize=21)
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\hist_90flood.eps",
          dpi=300, format="eps")


# TSS
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=df['TSS (mg/L)'], kde=False,
             hue=df["Community"], palette=['#ADD8E6', '#032180', '#5DC069', '#006E0D'],
             element="bars", legend=True)
# ax.set_title("Distribution of Total Suspended Solids", fontsize=24)
ax.set_xlabel('TSS (mg/L)', fontsize=21)
ax.set_ylabel("Count", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

f.legend(fontsize=21)
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\hist_TSS.eps",
          dpi=300, format="eps")


########## Complimenting scatter plots
# Tidal plot
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.scatterplot(data=df, x='Tidal Amplitude (cm)', y='Accretion Rate (mm/yr)',
                hue='90th Percentile Flood Depth (cm)',
                size='90th Percentile Flood Depth (cm)')
ax.set_xlabel('Tidal Amplitude (cm)', fontsize=21)
ax.set_ylabel('Accretion Rate (mm/yr)', fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='90th Percentile Flood Depth (cm)', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\scatter_tidal_flood_accretion.eps",
            dpi=300, format="eps")


# NDVI plot
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.scatterplot(data=df, x='NDVI', y='Accretion Rate (mm/yr)',
                hue='Soil Porewater Salinity (ppt)',
                size='Soil Porewater Salinity (ppt)')
ax.set_xlabel('NDVI', fontsize=21)
ax.set_ylabel('Accretion Rate (mm/yr)', fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Soil Porewater Salinity (ppt)', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\scatter_ndvi_salinity_accretion.eps",
            dpi=300, format="eps")


# Salinity plot
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.scatterplot(data=df, x='Soil Porewater Salinity (ppt)', y='Accretion Rate (mm/yr)',
                hue='NDVI',
                size='NDVI')
ax.set_xlabel('Soil Porewater Salinity (ppt)', fontsize=21)
ax.set_ylabel('Accretion Rate (mm/yr)', fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='NDVI', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\scatter_salinity_ndvi_accretion.eps",
            dpi=300, format="eps")


# 90th percentile flood plot
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.scatterplot(data=df, x='90th Percentile Flood Depth (cm)', y='Accretion Rate (mm/yr)',
                hue='Tidal Amplitude (cm)',
                size='Tidal Amplitude (cm)')
ax.set_xlabel('90th Percentile Flood Depth (cm)', fontsize=21)
ax.set_ylabel('Accretion Rate (mm/yr)', fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Tidal Amplitude (cm)', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\scatter_flood_tide_accretion.eps",
            dpi=300, format="eps")


# TSS plot
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.scatterplot(data=df, x='TSS (mg/L)', y='Accretion Rate (mm/yr)',
                hue='Bulk Density (g/cm3)',
                size='Bulk Density (g/cm3)')
ax.set_xlabel('TSS (mg/L)', fontsize=21)
ax.set_ylabel('Accretion Rate (mm/yr)', fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Bulk Density (g/cm3)', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\results_GPR\\scatter_tss_density_accretion.eps",
            dpi=300, format="eps")

