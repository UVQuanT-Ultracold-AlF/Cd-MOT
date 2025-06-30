#!/bin/sh
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=256:mem=920Gb

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp $HOME/Cd_MOT_paper/Cd-MOT/AlF_temp_MCWF.py $TMPDIR

cd $TMPDIR

$HOME/anaconda3/bin/python AlF_temp_MCWF.py
cp $TMPDIR/out2.npz $HOME/Cd_MOT_paper/Cd-MOT/out_AlF_temp_10ms_2photon_-4G_25-06-19.npz