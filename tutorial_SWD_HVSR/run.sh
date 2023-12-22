#!/bin/bash
###############################################################
#  Bourne shell script for submitting a parallel Open-MPI job #
#  to the SGE queue using the qsub command.                   #
#  (qsub, qdel, qstat, qconf)                                 #
###############################################################
#  Usage: 
#   (1) Submitting job: qsub run.sh 
#   (2) Killing job: qdel (job id)
#   (3) Checking job: qstat -f
###############################################################
#  Remarks: A line beginning with # is a comment.             #
#           A line beginning with #$ is a SGE directive.      #
#            SGE directives must come first; any directives   #
#            after the first executable statement are ignored.#
###############################################################
#   The SGE directives
###############################################################
# pe request (check available "pe"s by "qconf -spl")
# our Job name 

#$ -pe fill_up 32  # core number
#$ -N GW_syn_mode
#$ -S /bin/bash
#$ -q all.q
#$ -V
#$ -cwd

echo ------------------------------------------------------
echo 'This job is allocated on '${NSLOTS}' cpu(s)'
echo ------------------------------------------------------
echo SGE: qsub is running on $SGE_O_HOST
echo SGE: executing queue is $QUEUE
echo SGE: working directory is $SGE_O_WORKDIR
echo SGE: execution mode is $ENVIRONMENT
echo SGE: job identifier is $JOB_ID
echo SGE: job name is $JOB_NAME
echo SGE: node file is $TMPDIR/machines
echo SGE: current home directory is $SGE_O_WORKDIR
echo ------------------------------------------------------
#######################################################
### openmpi 1.4.1 (w/ Intel compiler)
#######################################################
#
### MPI_HOME=/opt/mpi/intel-12.1/openmpi-1.4.4
# MPI_HOME=/opt/intel/oneapi/mpi/2021.3.0/bin/

#
MPI_EXEC=mpirun
cd $SGE_O_WORKDIR
currentdir=`pwd`
echo $currentdir
PATH=$PATH:$HOME/bin:"/opt/intel/oneapi/intelpython/latest/bin/:/opt/intel/oneapi/mpi/2021.3.0/bin/:$PATH"
#

#cd /home/hscheon/specf/SHAKEMAP/3D/129.19_35.76/
#mpirun -hostfile $TMPDIR/machines xdecompose_mesh $NSLOTS MESH/ OUTPUT_FILES/DATABASES_MPI/
#cd ..

# cp in_out_files/OUTPUT_FILES/output_mesher.txt in_out_files/OUTPUT_FILES/output_meshfem3D.txt
#cp DATA/Par_file OUTPUT_FILES/
#cp DATA/CMTSOLUTION OUTPUT_FILES/
#cp DATA/STATIONS OUTPUT_FILES/

#cd $currentdir/bin
#mpirun -hostfile $TMPDIR/machines  -np $NSLOTS xgenerate_databases

python3 tutorialhunt.py
#mpirun -hostfile $TMPDIR/machines -np $NSLOTS ./bashscript_forMPI.sh  # instead of "xspecfem3d" ./your script filename
#cd ..

#rm -rf in_out_files/DATABASES_MPI


exit
