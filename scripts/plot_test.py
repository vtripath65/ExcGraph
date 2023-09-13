import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#######################################################
#  Plots the predicted data for test set.             #
#  Predicted results are plotted against target       #
#  values with or without labels.                     #
#                      Vikrant Tripathy (08/01/2023)  #
#######################################################

def plot_test(Xdata,Ydata,Xlabel,Ylabel,Title,Sol,RegCol,fig,ax):
  y=[]
  z = np.polyfit(Xdata,Ydata,1)
  p = np.poly1d(z)
  for i in range(len(Xdata)):
    y.append(z[0]*Xdata[i]+z[1])
  slope,intercept,r_value,p_value,std_err=stats.linregress(Ydata,y)
  r2value=r_value**2
  if z[1]>=0:
    text = "y={0:0.3f}x+{1:0.3f}\n${corr}$ = {2:0.3f}".format(z[0],z[1],r2value,corr = "\mathregular{R^2}")
  else:
    text = "y={0:0.3f}x{1:0.3f}\n${corr}$ = {2:0.3f}".format(z[0],z[1],r2value,corr = "\mathregular{R^2}")
  print(Title)

# Only choose solvents with atleast 40 data points
  sol_list=['Dichloromethane','Acetonitrile','Methanol','TetraHydroFuran','Chloroform','Toluene','Ethanol','DiMethylSulfoxide','cyclohexane','EthylEthanoate','Water','n,n-DiMethylFormamide','n-hexane','1,4-Dioxane','Acetone','Benzene','2-Propanol','DiethylEther','1-Butanol','1-Propanol','Heptane','CarbonTetraChloride','DiButylEther','MethylCycloHexane','2,2,2-TriFluoroEthanol']

  (datax,datay)=({},{})
  for i in sol_list:
    datax[i]=[]
    datay[i]=[]
  datax['Misc']=[]
  datay['Misc']=[]

  Xdata1=[]
  Ydata1=[]
  for i in range(len(Xdata)):
    if Sol[i] in sol_list:
      datax[Sol[i]].append(Xdata[i])
      datay[Sol[i]].append(Ydata[i])
    else:
      datax['Misc'].append(Xdata[i])
      datay['Misc'].append(Ydata[i])

  ax.scatter(datax['Dichloromethane'],datay['Dichloromethane'],color=(0.33, 0.33, 0.33),s=8.0,alpha=0.5,label='Dichloromethane')
  ax.scatter(datax['Acetonitrile'],datay['Acetonitrile'],color=(0.66, 0.33, 0.33),s=8.0,alpha=0.5,label='Acetonitrile')
  ax.scatter(datax['Methanol'],datay['Methanol'],color=(1.0, 0.33, 0.33),s=8.0,alpha=0.5,label='Methanol')
  ax.scatter(datax['TetraHydroFuran'],datay['TetraHydroFuran'],color=(0.33, 0.66, 0.33),s=8.0,alpha=0.5,label='TetraHydroFuran')
  ax.scatter(datax['Chloroform'],datay['Chloroform'],color=(0.66, 0.66, 0.33),s=8.0,alpha=0.5,label='Chloroform')
  ax.scatter(datax['Toluene'],datay['Toluene'],color=(1.0, 0.66, 0.33),s=8.0,alpha=0.5,label='Toluene')
  ax.scatter(datax['Ethanol'],datay['Ethanol'],color=(0.33, 1.0, 0.33),s=8.0,alpha=0.5,label='Ethanol')
  ax.scatter(datax['DiMethylSulfoxide'],datay['DiMethylSulfoxide'],color=(0.66, 1.0, 0.33),s=8.0,alpha=0.5,label='DiMethylSulfoxide')
  ax.scatter(datax['cyclohexane'],datay['cyclohexane'],color=(1.0, 1.0, 0.33),s=8.0,alpha=0.5,label='cyclohexane')
  ax.scatter(datax['EthylEthanoate'],datay['EthylEthanoate'],color=(0.33, 0.33, 0.66),s=8.0,alpha=0.5,label='EthylEthanoate')
  ax.scatter(datax['Water'],datay['Water'],color=(0.66, 0.33, 0.66),s=8.0,alpha=0.5,label='Water')
  ax.scatter(datax['n,n-DiMethylFormamide'],datay['n,n-DiMethylFormamide'],color=(1.0, 0.33, 0.66),s=8.0,alpha=0.5,label='n,n-DiMethylFormamide')
  ax.scatter(datax['n-hexane'],datay['n-hexane'],color=(0.33, 0.66, 0.66),s=8.0,alpha=0.5,label='n-hexane')
  ax.scatter(datax['1,4-Dioxane'],datay['1,4-Dioxane'],color=(0.66, 0.66, 0.66),s=8.0,alpha=0.5,label='1,4-Dioxane')
  ax.scatter(datax['Acetone'],datay['Acetone'],color=(1.0, 0.66, 0.66),s=8.0,alpha=0.5,label='Acetone')
  ax.scatter(datax['Benzene'],datay['Benzene'],color=(0.33, 1.0, 0.66),s=8.0,alpha=0.5,label='Benzene')
  ax.scatter(datax['2-Propanol'],datay['2-Propanol'],color=(0.66, 1.0, 0.66),s=8.0,alpha=0.5,label='2-Propanol')
  ax.scatter(datax['DiethylEther'],datay['DiethylEther'],color=(1.0, 1.0, 0.66),s=8.0,alpha=0.5,label='DiethylEther')
  ax.scatter(datax['1-Butanol'],datay['1-Butanol'],color=(0.33, 0.33, 1.0),s=8.0,alpha=0.5,label='1-Butanol')
  ax.scatter(datax['1-Propanol'],datay['1-Propanol'],color=(0.66, 0.33, 1.0),s=8.0,alpha=0.5,label='1-Propanol')
  ax.scatter(datax['Heptane'],datay['Heptane'],color=(1.0, 0.33, 1.0),s=8.0,alpha=0.5,label='Heptane')
  ax.scatter(datax['CarbonTetraChloride'],datay['CarbonTetraChloride'],color=(0.33, 0.66, 1.0),s=8.0,alpha=0.5,label='CarbonTetraChloride')
  ax.scatter(datax['DiButylEther'],datay['DiButylEther'],color=(0.66, 0.66, 1.0),s=8.0,alpha=0.5,label='DiButylEther')
  ax.scatter(datax['MethylCycloHexane'],datay['MethylCycloHexane'],color=(1.0, 0.66, 1.0),s=8.0,alpha=0.5,label='MethylCycloHexane')
  ax.scatter(datax['2,2,2-TriFluoroEthanol'],datay['2,2,2-TriFluoroEthanol'],color=(0.33, 1.0, 1.0),s=8.0,alpha=0.5,label='2,2,2-TriFluoroEthanol')
  ax.scatter(datax['Misc'],datay['Misc'],color=(0.66, 1.0, 1.0),s=8.0,alpha=0.5,label='Miscellaneous')
  ax.plot(Xdata,p(Xdata),RegCol)
  lims=np.asarray(range(70))
  ax.plot(lims/10,lims/10,'r.')
  ax.text(0.05, 0.95, text,transform=plt.gca().transAxes,fontsize=14, fontweight='bold', verticalalignment='top')
  MAE = 'MAE = {0:0.3f}'.format(np.mean(np.abs(Xdata-Ydata)))
  ax.text(6.5, 0.05,  MAE, fontsize=14, fontweight='bold', verticalalignment='bottom', horizontalalignment='right')

  plt.xticks(fontsize=12,fontweight='bold')
  plt.yticks(fontsize=12,fontweight='bold')
  ax.set_xlabel(Xlabel, fontsize=12, fontweight='bold')
  ax.set_ylabel(Ylabel, fontsize=12, fontweight='bold')
  ax.set_title(Title, fontsize=12, fontweight='bold')

  print("minimum error: {0:0.3f}     maximum error: {1:0.3f}".format(np.min(Ydata-Xdata),np.max(Ydata-Xdata)))

  mae=np.mean(np.abs(Xdata-Ydata))
  mse=np.mean(Ydata-Xdata)
  print('MAE: {0:0.3f}'.format(mae))
  print('MSE: {0:0.3f}'.format(mse))
  Ydata1=(Ydata-mse)
  mad=np.mean(np.abs(Ydata1-Xdata))
  print('MAD: {0:0.3f}'.format(mad))
  print('minimum of corrected error: {0:0.3e}'.format(np.mean(Ydata1-Xdata)))

