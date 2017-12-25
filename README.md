# eBDIMS
Source code for the elastic network driven Brownian Dynamics Importance Sampling simulation method (eBDIMS).

For more information about the eBDIMS method please read the following publication:

Orellana, L. et al. Prediction and validation of protein intermediate states from structurally rich ensembles and coarse-grained simulations. Nat. Commun. 7:12575 doi: 10.1038/ncomms12575 (2016).

This repository contains the source code which is used to run simulations on the web server at https://login.biophysics.kth.se/eBDIMS/. Note that most of the error-handling has been implemented within the web server and not in the source code. Users who wish more applications related to user input or analysis are referred to run simulations through the web server.

This code runs with shared-memory parallelism (OpenMP) and can be compiled and executed with gcc:

    gcc -o <output> -fopenmp eBDIMS_parallel.c -lm
    ./<output> <start pdb file name> <target pdb file name> <cutoff> <mode> <number of unbiased steps>
