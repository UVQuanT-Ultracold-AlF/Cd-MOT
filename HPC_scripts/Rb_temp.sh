#!/bin/sh
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=256:mem=400Gb
#PBS -J 0-21

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp -r $HOME/Cd_MOT_paper/Cd-MOT/Rb_temp_full.py $TMPDIR

cd $TMPDIR

DETUNINGS=("-100" "-90" "-80" "-70" "-60" "-50" "-40" "-30" "-20" "-10" "0" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

$HOME/anaconda3/bin/python Rb_temp_full.py ${DETUNINGS[${PBS_ARRAY_INDEX}]}
cp $TMPDIR/out.npz $HOME/Cd_MOT_paper/Cd-MOT/out_Rb_temp_whole_${DETUNINGS[${PBS_ARRAY_INDEX}]}.npz
