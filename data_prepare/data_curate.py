#!/usr/bin/python
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import numpy as np
import re, os
from glob import glob

# Get no of unique chromophores and solvents and combinations
def get_unique(Dict):
  values=list(Dict.values())
  keys=list(Dict.keys())
  sol_subset=[i[1] for i in values]
  print(len(sol_subset))
  print(len(set(sol_subset)))
  mol_subset=[i[0] for i in values]
  print(len(set(mol_subset)))

  with open('unique_chromophores.csv','w') as fh:
    for i in set(mol_subset):
      fh.write(i+'\n')

# for sorting
def myfunc(e):
  return e[1]

# look for how many molecules different solvents have
def occurrence_sols(Dict):
  values=list(Dict.values())
  sol=[i[1] for i in values]
  sol_set=set(sol)
  set10,set20,set30,set40,set50=0,0,0,0,0
  N10,N20,N30,N40,N50=0,0,0,0,0
  count=[]
  for i in sol_set:
    count.append((i,sol.count(i)))
    if sol.count(i) >= 50:
      N50=N50+sol.count(i)
      N40=N40+sol.count(i)
      N30=N30+sol.count(i)
      N20=N20+sol.count(i)
      N10=N10+sol.count(i)
      set50=set50+1
      set40=set40+1
      set30=set30+1
      set20=set20+1
      set10=set10+1
    elif sol.count(i) >= 40:
      N40=N40+sol.count(i)
      N30=N30+sol.count(i)
      N20=N20+sol.count(i)
      N10=N10+sol.count(i)
      set40=set40+1
      set30=set30+1
      set20=set20+1
      set10=set10+1
    elif sol.count(i) >= 30:
      N30=N30+sol.count(i)
      N20=N20+sol.count(i)
      N10=N10+sol.count(i)
      set30=set30+1
      set20=set20+1
      set10=set10+1
    elif sol.count(i) >= 20:
      N20=N20+sol.count(i)
      N10=N10+sol.count(i)
      set20=set20+1
      set10=set10+1
    elif sol.count(i) >= 10:
      N10=N10+sol.count(i)
      set10=set10+1
  print(set10,set20,set30,set40,set50)
  print(N10,N20,N30,N40,N50)
  count.sort(reverse=True,key=myfunc)
  print(count)

def remove_approximate_sol(Dict,Sol):
  for x,i in enumerate(Sol):
    if i=='CC(=O)C(C)(C)C':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CC(O)CO':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCCCCCCCCO':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CC(C)(C)O':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='OC1CCCCC1':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='COCCOC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCOCCCCC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCOC(=O)c1ccccc1':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCl':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CC1CCCO1':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCCCCCCCCCC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCC(C)C':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCC(C)(C)O':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCOCCC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCCCCCCCO':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCCCCCCCCCCCCCCC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='C1CCC2CCCCC2C1':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i=='CCC(C)CC':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]

def remove_radicals(Dict):
  del Dict[19658]
  del Dict[19659]
  del Dict[19660]
  del Dict[19661]
  del Dict[19662]
  del Dict[921]
  del Dict[922]
  del Dict[923]
  del Dict[924]
  del Dict[929]
  del Dict[930]
  del Dict[931]
  del Dict[932]

def remove_duplicate(Dict,Ems):
  values=list(Dict.values())
  keys=list(Dict.keys())
  with open('duplicate.csv','w') as fh:
    for x,i in enumerate(values):
      if values.count(i) > 1:
        if not os.path.isfile('mom-incl-triplet/mom/'+str(keys[x])+'/'+str(keys[x])+'.gjf'):
          fh.write(str(keys[x])+','+values[x][0]+','+values[x][1]+','+str(Ems[keys[x]-1])+'\n')
          del Dict[keys[x]]

def remove_large(Dict,cutoff):
  values=list(Dict.values())
  keys=list(Dict.keys())
  for x,i in enumerate(values):
    mol=Chem.MolFromSmiles(i[0])
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() > cutoff:
      del Dict[keys[x]]

def remove_unknown_sol(Dict,Sol):
  for x,i in enumerate(Sol):
    if i != 'gas' and i != 'CCCCOP(=O)(OCCCC)OCCCC':
      if re.search('\(',i) or re.search('/',i):
        mol=Chem.MolFromSmiles(i)
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() > 35:
          if x+1 in list(Dict.keys()):
            del Dict[x+1]
    if re.search('Si',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('\[B-\]',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('B\(',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('NH+',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('N#C/',i) or re.search('\(C#N\)',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('p',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if len(re.findall('s',i)) > 1:
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('3',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i != 'C1CCC2CCCCC2C1' and re.search('2',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if re.search('P',i) and re.search('N',i):
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i == 'CN(C)C(=O)N(C)C' or i == 'C1COCCN1' or i == 'CCCCCCCCCCCCCCCCCl' or i == 'OC(C(F)(F)F)C(F)(F)F' or i == 'CN1CCCC1=O' or i == 'C=CC#N':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]
    if i == 'OCCOCCO' or i == 'OCC(O)CO' or i == 'O=C1CCCO1' or i == 'Oc1ccccc1' or i == 'OCCOc1ccccc1' or i == 'CC(=O)OC(C)=O' or i == 'CC1COC(=O)O1':
      if x+1 in list(Dict.keys()):
        del Dict[x+1]

def convergence_issues(Dict):
  list1=glob('mom-incl-triplet/mom/*/')
  list1=[re.sub('mom-incl-triplet/mom/','',x) for x in list1]
  list1=[re.sub('/','',x) for x in list1]
  for i in list(Dict.keys()):
    if str(i) not in list1:
      del Dict[i]
#convergence while including triplet
  del Dict[11990]
  del Dict[12543]
  del Dict[15793]
  del Dict[4870]
  del Dict[6120]

def identify(Tup):
  Mol=[Chem.MolFromSmarts(Chem.MolToSmiles(Chem.MolFromSmiles(x.strip()),canonical=True)) for x in open('dye_types_large_dataset.csv','r').readlines()]
  bodipy=Mol[0]
  triphenylamine=Mol[1]
  carbazole=Mol[2]
  anthracene=Mol[3]
  stilbene=Mol[4]
  Mol[5]=Chem.MolFromSmarts('[C,c]([C,H])[N;!+]C=C/C=C/C=C/C=[N+]([C,H])[C,c]')
  cyanine=Mol[5]
  coumarin=Mol[6]
  pyrene=Mol[7]
  naphthalimide=Mol[8]
  benzothiazole=Mol[9]
  Mol[10]=Chem.MolFromSmarts('[c,n]1[c,n][c,n]2[c,n][c,n][c,n][c,n]3[c,n]4[c,n][c,n][c,n][c,n]5[c,n][c,n][c,n][c,n]([c,n]([c,n]1)[c,n]23)[c,n]54')
  perylene=Mol[10]
  Mol[11]=Chem.MolFromSmarts('[X3&B]')
  organoboron=Mol[11]
  benzoxazole=Mol[12]
  rhodamine=Mol[13]
  phenoxazine=Mol[14]
  helicene=Mol[15]
  azobenzene=Mol[16]
  Mol[17]=Chem.MolFromSmarts('C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH,s]5)C=C4)[nH,s]3')
  porphyrin=Mol[17]
  fluorescein=Mol[18]
  cyanine1=Chem.MolFromSmarts('[C,c]([C,H])[N;!+]C=C/C=C/C=[N+]([C,H])[C,c]')
  cyanine2=Chem.MolFromSmarts('[C,c]([C,H])[N;!+]C=C/C=C/C=C/C=C/C=[N+]([C,H])[C,c]')

  mol = Chem.MolFromSmiles(Tup[1][0])

  count=0
  for x in Mol:
    if mol.HasSubstructMatch(x):
      count=count+1
      xname = [k for k,v in locals().items() if v == x][0]
      with open('chromophore_wise_subset/'+xname+'.txt','a') as fh:
        fh.write(str(Tup[0])+'\n')

  if count > 1:
    print(Tup[1][0]+'    '+str(count))

  if mol.HasSubstructMatch(cyanine1) or mol.HasSubstructMatch(cyanine2):
    with open('chromophore_wise_subset/cyanine.txt','a') as fh:
      fh.write(str(Tup[0])+'\n')

if __name__ == '__main__':
  df = pd.read_csv('DB_for_chromophore_Sci_Data_rev02_rem_anions.csv')
  Index=df['Tag'].values.tolist()
  Ems=df['Emission max (nm)'].values.tolist()
  Abs=df['Absorption max (nm)'].values.tolist()
  Chromophore=df['Chromophore'].values.tolist()
  Chromophore=[Chem.CanonSmiles(m) for m in Chromophore]
  Sol=df['Solvent'].values.tolist()
  Sol=[sub.replace('([2H])','') for sub in Sol]
  Sol=[sub.replace('[2H]','') for sub in Sol]
  temp=Sol
  Sol=[]
  for i in temp:
    if i != 'gas':
      Sol.append(Chem.CanonSmiles(i))
    else:
      Sol.append(i)
#
#  We assuming that any data with ems value is good data for us
#  This is due to the stable excited state. The missing Abs value
#  may be due to it being not reported.
#
  with_ems={}
  without_ems={}
  No_opt_data={}
  for i,x in enumerate(Ems):
    if str(x) != 'nan':
      with_ems[Index[i]]=(str(Chromophore[i]),str(Sol[i]))
    elif str(Abs[i]) != 'nan':
      without_ems[Index[i]]=(str(Chromophore[i]),str(Sol[i]))
    else:
      No_opt_data[Index[i]]=(str(Chromophore[i]),str(Sol[i]))

#
#  I am filtering the data points to get good data points.
#  I am printing the unique data points after any such filtration process.
#

  get_unique(with_ems)
  remove_duplicate(with_ems,Ems)
  get_unique(with_ems)
  remove_large(with_ems,80)
  get_unique(with_ems)
  remove_unknown_sol(with_ems,Sol)
  remove_approximate_sol(with_ems,Sol)
  get_unique(with_ems)
# remove C70
  del with_ems[7145]
  get_unique(with_ems)
  remove_radicals(with_ems)
  get_unique(with_ems)
  convergence_issues(with_ems)
  get_unique(with_ems)
  occurrence_sols(with_ems)

#
#  Identify chromophore types
#

  pool = ThreadPool(30)
  results = pool.map(identify, with_ems.items())
  pool.close()
  pool.join()
