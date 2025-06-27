#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=256:mem=920Gb

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp -r $HOME/Cd_MOT_paper/Cd-MOT/* $TMPDIR

cd $TMPDIR

$HOME/anaconda3/bin/python AlF_diff_vel.py
cp $TMPDIR/out.npz $HOME/Cd_MOT_paper/Cd-MOT/out_AlF_d_cap_vel.npz