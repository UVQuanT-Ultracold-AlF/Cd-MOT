#!/bin/sh
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=256:mem=400Gb

#module load anaconda3/personal

. $HOME/.bashrc

PWD=${pwd}

cp -r $HOME/Cd_MOT_paper/Cd-MOT/Rb_temp.py $TMPDIR

cd $TMPDIR
ARR_INDEX=0
DETUNINGS=("-100" "-90" "-80" "-70" "-60" "-50" "-40" "-30" "-20" "-10" "0" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

$HOME/anaconda3/bin/python Rb_temp.py ${DETUNINGS[${ARR_INDEX}]}
cp $TMPDIR/out.npz $HOME/Cd_MOT_paper/Cd-MOT/out_Rb_temp${DETUNINGS[${ARR_INDEX}]}.npz
