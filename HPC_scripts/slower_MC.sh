#!/bin/sh
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=256:mem=920Gb

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp -r $HOME/Cd_MOT_paper/Cd-MOT/* $TMPDIR

cd $TMPDIR

$HOME/anaconda3/bin/python slower_scan_MC.py 10000 256
cp $TMPDIR/out.npy $HOME/Cd_MOT_paper/Cd-MOT/out.npy
