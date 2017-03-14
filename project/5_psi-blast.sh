# run a psi-blast


#change directory to uniref90 
export BLASTDB=/local_uniref/uniref/uniref90

for seq in /home/u2206/Desktop/PSI_BLAST_TM2/* /home/u2206/Desktop/PSI_BLAST_TM2; do



if [ ! -f $output_directory/$base.psi ]; then
	echo "Running psiblast on $seq at $(date)..."
	time psiblast -query $seq -db uniref90.db -num_iterations 3 -evalue 0.001 -out $seq.psiblast -out_ascii_pssm $seq.pssm -num_threads 8
	echo "Finished running psiblast on $seq at $(date)."
fi
done


echo 'PSI-BLAST run is complete'


cd ~/Desktop/PSI_BLAST_TM2/
mv *.psiblast ~/Desktop/psiblast_end/
mv *.pssm ~/Desktop/pssm/
