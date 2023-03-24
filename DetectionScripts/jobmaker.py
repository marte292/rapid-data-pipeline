import os
import sys

if __name__ == '__main__':

    assert(len(sys.argv)==4)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    _,rootdir,outputdir,email = sys.argv
    runNum = rootdir.split('_')[1]

    jobFolder = 'run{}jobfiles'.format(runNum)
    if jobFolder not in os.listdir():
        os.mkdir(jobFolder)
        os.popen('cp runJobs.sh {}/runJobs.sh'.format(jobFolder))

    outFolder = 'run{}'.format(runNum)
    if outFolder not in os.listdir(outputdir):
        os.mkdir(os.path.join(outputdir,outFolder))

    list_subfolders_with_paths = [f.path for f in os.scandir(rootdir) if f.is_dir()]

    for i,folder in enumerate(list_subfolders_with_paths):
        f = open(os.path.join(jobFolder,"job"+str(i+1)+".txt"),"w+")
        f.write("#!/bin/bash \n")
        f.write("#SBATCH -J Run" + str(runNum).zfill(2) + str(i+1) + "\n")
        f.write("#SBATCH -o Run" + str(runNum).zfill(2) + str(i+1) + ".o \n")
        f.write("#SBATCH -e Run" + str(runNum).zfill(2) + str(i+1) + ".e \n")
        f.write("#SBATCH -p rtx \n")
        f.write("#SBATCH -N 1 \n")
        f.write("#SBATCH -n 1 \n")
        f.write("#SBATCH -t 12:59:00 \n")
        f.write("#SBATCH --mail-type=all \n")
        f.write("#SBATCH -A BCS20006 \n")
        f.write("#SBATCH --mail-user={} \n".format(email))
        f.write("python3 $HOME/src/mmdet/tools/CorrectAndDetect.py $HOME/src/mmdet/configs/elephant/crowdhuman/cascade_hrnet.py $HOME/src/mmdet/models_pretrained/epoch_19.pth.stu " +\
          folder + " --out {}/run{}/output".format(outputdir,runNum) + str(i+1) + ".pkl")
        f.close()