import contextlib
import datetime
import os
import sys

import pandas as pd

from utils.ForHiggsTools import ProcessDataml, ReadFortranOutputFilesToDF


path_fortran = ''
path_python = ''

pd.set_option('max_colwidth',None)
#pd.set_option('display.max_rows',None)

current_time_string = lambda : str(datetime.datetime.now().isoformat())


def NumLinesTxt(path):
	with open(path, 'r') as file:
		res = len(file.readlines())
	return res

############################################################################
#   write fortran input
############################################################################

TF = lambda x : "T" if x else "F"

############################################################################
#   run HT
############################################################################

def run_HiggsTools(
	path=path_fortran,
	neutralCS=True,
	chargedCS=True,
	Couplings=True,
	neutralW=True,
	chargedW=True,
	flags=False,
	Do_Chisq=False,
):


        # run
	df=ReadFortranOutputFilesToDF(path)

	res = ProcessDataml(
		df,
		neutralCS=neutralCS,
		chargedCS=chargedCS,
		Couplings=Couplings,
		neutralW=neutralW,
		chargedW=chargedW,
		flags=flags,
		Do_Chisq=Do_Chisq,
)	
		

#	print(res)
	#res = ProcessData(df,neutralCS=neutralCS,chargedCS=chargedCS,Couplings=Couplings,neutralW=neutralW,chargedW=chargedW,flags=flags)	
		
	# print
	#print()
#	print(res[["index","HBres"]+['limits_{}'.format(scalar) for scalar in ['h1','h2','h3','a1','a2','Hp1','Hp2']]+['key','description']])		
	#print()

	return res

