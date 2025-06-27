#!/bin/sh
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=256:mem=920Gb

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp $HOME/Cd_MOT_paper/Cd-MOT/CaF_temp_MCWF.py $TMPDIR

cd $TMPDIR

$HOME/anaconda3/bin/python CaF_temp_MCWF.py
cp $TMPDIR/out2.npz $HOME/Cd_MOT_paper/Cd-MOT/out_CaF_temp_Harvard_sp_0mK_500um_s2_25-06-04_flipped_da_db.npz