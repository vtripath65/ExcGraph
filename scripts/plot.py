import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re

#######################################################
#  Plots the predicted data for PhotochemCAD dataset  #
#  and its subsets with common and not common dye     #
#  molecules with the large (12318) dataset.          #
#  Predicted results are plotted against target       #
#  values with or without labels.                     #
#                      Vikrant Tripathy (08/01/2023)  #
#######################################################

def plot(Xdata,Ydata,Xlabel,Ylabel,Title,Name,RegCol,Annotate,fig,ax):
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

  print('minimum error: {0:0.3f}     maximum error: {1:0.3f}'.format(np.min(Ydata-Xdata),np.max(Ydata-Xdata)))
  MAE = 'MAE = {0:0.3f}'.format(np.mean(np.abs(Xdata-Ydata)))
  print('MAE: {0:0.3f}'.format(np.mean(np.abs(Xdata-Ydata))))
  print('MSE: {0:0.3f}'.format(np.mean(Ydata-Xdata)))
  Ydata1=(Ydata-z[1])/z[0]
  print('MAD: {0:0.3f}'.format(np.mean(np.abs(Xdata-Ydata1))))
  print('minimum of corrected error: {0:0.3e}'.format(np.mean(Ydata1-Xdata)))

  Type=['I','F','L','H','K','P','E','O','B','C','R','Q','S','T','G','D','N','A','M']

  (datax,datay)=({},{})
  for i in Type:
    datax[i]=[]
    datay[i]=[]

  for i in range(len(Xdata)):
    for j in Type:
      if re.search(j,Name[i]):
        datax[j].append(Xdata[i])
        datay[j].append(Ydata[i])
        break

  ax.scatter(datax['I'],datay['I'],c="limegreen",marker="o")
  ax.scatter(datax['F'],datay['F'],c="orange",marker="1")
  ax.scatter(datax['L'],datay['L'],c="purple",marker="s")
  ax.scatter(datax['H'],datay['H'],c="black",marker="d")
  ax.scatter(datax['K'],datay['K'],c="grey",marker="8")
  ax.scatter(datax['P'],datay['P'],c="gold",marker="p")
  ax.scatter(datax['E'],datay['E'],c="darkgreen",marker="h")
  ax.scatter(datax['O'],datay['O'],c="darkorange",marker="H")
  ax.scatter(datax['B'],datay['B'],c="brown",marker="*")
  ax.scatter(datax['C'],datay['C'],c="blue",marker="x")
  ax.scatter(datax['R'],datay['R'],c="olive",marker=">")
  ax.scatter(datax['Q'],datay['Q'],c="darkblue",marker="X")
  ax.scatter(datax['S'],datay['S'],c="turquoise",marker="+")
  ax.scatter(datax['T'],datay['T'],c="tan",marker="<")
  ax.scatter(datax['G'],datay['G'],c="red",marker="^")
  ax.scatter(datax['D'],datay['D'],c="magenta",marker="v")
  ax.scatter(datax['N'],datay['N'],c="cyan",marker="P")
  ax.scatter(datax['A'],datay['A'],c="violet",marker="4")
  ax.scatter(datax['M'],datay['M'],c="maroon",marker="D")
  ax.plot(Xdata,p(Xdata),RegCol)
  lims=np.asarray(range(70))
  if Xlabel != "Max similarity":
    ax.plot(lims/10,lims/10,'r.')
    ax.text(0.05, 0.95, text,transform=plt.gca().transAxes,fontsize=14, fontweight='bold', verticalalignment='top')
  else:
    ax.text(0.6, 0.95, text,transform=plt.gca().transAxes,fontsize=14, fontweight='bold', verticalalignment='top')
  ax.text(6.5, 0.05,  MAE,fontsize=14, fontweight='bold', verticalalignment='bottom', horizontalalignment='right')

  if Annotate:
    for i, txt in enumerate(Name):
      ax.annotate(txt,(Xdata[i],Ydata[i]),fontsize='smaller')

  plt.xticks(fontsize=12,fontweight='bold')
  plt.yticks(fontsize=12,fontweight='bold')
  ax.set_xlabel(Xlabel, fontsize=12, fontweight='bold')
  ax.set_ylabel(Ylabel, fontsize=12, fontweight='bold')
  ax.set_title(Title, fontsize=12, fontweight='bold')
  plt.show()

