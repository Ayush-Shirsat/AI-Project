# AI-Project
CS 640: AI 

Steps for SCC:
1) ```ssh username@scc1.bu.edu``` (login)
2) navigate to cs640grp. Make a folder of your name. Upload all files you need. (should be default if AI was the first time you used scc)
3) ```qrsh -P cs640grp -l gpus=1 -l gpu_c=6 -l h_rt=24:00:00``` (request a GPU)
4) ```module load python/3.6.2```
5) ```module load opencv/3.3.0```
6) ```module load tensorflow/r1.10```
7) ```jupyter notebook``` (Will have to setup if using first time: http://www.bu.edu/tech/support/research/software-and-programming/common-languages/python/jupyter/). Default browser is mozilla and be ready to experience lag :)
