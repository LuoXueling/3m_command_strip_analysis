# 3M Command Strip fracture mechanics analysis

This repository contains code and data for the paper [*How do 3M Command™ strips work? A fracture mechanics approach*, *Soft Matter* (2026), by Xue-Ling Luo, Nikolaos Bouklas, and Chung-Yuen Hui](https://pubs.rsc.org/sm/article/doi/10.1039/d6sm00672h/1365960/How-do-3M-CommandT-strips-work-A-fracture), including the code to perform finite element analysis for 3M Command™ Strips and the data used to generate the figures in the paper. 

`tape.py` is the Abaqus script to automatically generate the finite element model, perform the analysis, and postprocess the results. `plots/commands.sh` contains all command lines to generate the figures in the paper, accompanied with raw data files `plots/*.csv` we obtained. 
