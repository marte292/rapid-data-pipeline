for ((i=$1;i<$2;i++))
do
  	sbatch "job$i.txt";
done