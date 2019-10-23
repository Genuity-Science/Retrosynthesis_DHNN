#!/bin/bash
now=$(date)
now=$(date +'%m_%d_%Y_%r' | sed 's/:/_/g' | sed 's/ /_/g')

echo $now

logfile="mapping_${now}.log"

if [ -e $logfile ]; then
	rm -f $logfile
fi

	#source /home/jbaylon/miniconda2/bin/activate /home/jbaylon/miniconda2/envs/rdkit-env 

	#/home/jbaylon/miniconda2/envs/rdkit-env/bin/python map_canonical_rxn_indigo_multiprocess.py >> $logfile &
	#/home/jbaylon/miniconda2/envs/rdkit-env/bin/python map_canonical_rxn_indigo_multiprocess.py | tee -a $logfile &

	#unbuffered output to track log file in real time!
#/Users/jbaylon/Scr/keras.tutorial/keras/bin/python -u highway_model_1024.py >> $logfile &
coolpython="/home/jbaylon/miniconda2/envs/molvae/bin/python"
$coolpython -u highway_model.py >> $logfile &
