////////////////////////////////////////////////////////////////////////////////
//    This program runs Elastic Network driven Brownian Dynamics              //
//    simultions of coarse-grained biomolecules with shared memory            //
//    parallelism. For more details about the method see the following        //
//    publication:                                                            //
//                                                                            //
//        Orellana, L. et al. Prediction and validation of protein            //
//        intermediate states from structurally rich ensembles and            //
//        coarse-grained simulations. Nat. Commun. 7:12575                    //
//        doi: 10.1038/ncomms12575 (2016).                                    //
//                                                                            //
//    Usage with gcc:                                                         //
//    gcc -o <output> -fopenmp eBDIMS_parallel.c -lm                          //
//    ./<output> <start pdb file name> <target pdb file name> <cutoff>        //
//    <mode> <number of unbiased steps>                                       //
//                                                                            //
//    NOTES: This code does not contain much error-handling of input data.    //
//           For more applications regarding input or analysis users are      //
//           referred to run simulations through the web server at            //
//           https://login.biophysics.kth.se/eBDIMS/                          //
//                                                                            //
//           It is important that the start and target pdb files contain      //
//           consistent information, only the spatial coordinates should      //
//           be different. This may require some pre-processing of the        //
//           input files.                                                     //
//                                                                            //
//    Written by Cathrine Bergh, Master's student at the Royal Institute      //
//    of Technology, Sweden, 2017.                                            //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define C 40.0                                 //Constant for Kovacs mode
#define CCART 6.0                              //Constant for mixed mode
#define CSEQ 60.0                              //Constant for mixed mode
#define DT 1.0E-15                             //Time step size
#define EX 6.0                                 //Exponent in mixed mode
#define INVMASS 6.947857E-11                   //Inverse of particle mass
#define KB 1.38065E-23                         //Boltzmann's constant
#define KLIN 10.0                              //Force constant in elastic network
#define PI 3.14159265358979                    //Pi
#define PMASS (100.0*1.67E-27)                 //Particle mass
#define POW (10E10)                            //10^10
#define RCONST 3.8                             //Constant for Kovacs mode
#define SCA 1.0                                //Scaling constant
#define SLIM 3                                 //Definition of nearest neighbour
#define SIGMA 1.0                              //Variance for gaussian distributed random numbers
#define SUBUNIT_CUTOFF 8.0                     //Cutoff distance between different subunits (chains)
#define T0 300.0                               //Temperature
#define TAU 0.4E-14                            //Time constant
#define TIME_LIMIT 86400                       //Wallclock time limit [seconds]

#define boltzSigma sqrt(KB*T0/PMASS)
#define eps (2*1.38E-23*T0*(PMASS/TAU))
#define c0 exp(-DT/TAU)
#define c1 ((1 - c0)*(TAU/DT))
#define c2 ((1 - c1)*(TAU/DT))
#define av c0
#define ar (c1*DT)
#define bv ((1 - c0)*(TAU/PMASS))
#define br (c2*DT*DT)/PMASS
#define cv (c0*sqrt(DT)*sqrt(eps)*pow(10,10)/PMASS)
#define cr ((1 - c0)*sqrt(DT*eps)*pow(10,10)*(TAU/PMASS))

//MERSENNE TWISTER PSEUDO-RANDOM NUMBER GENERATOR
//(Makoto Matsumoto and Takuji Nishimura, 1997)
typedef unsigned long uint32;

#define NMT            (624)                 // length of state vector
#define MMT            (397)                 // a period parameter
#define KMT            (0x9908B0DFU)         // a magic constant
#define MAX_RAND       (pow(2,32)-1)         // largest possible random number
#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v

static uint32   state[NMT+1];     // state vector + 1 extra to not violate ANSI C
static uint32   *next;            // next random value is computed from here
static int      left = -1;        // can *next++ this many times before reloading

//Create a random seed
void seedMT(uint32 seed){
  register uint32 x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
  register int    j;

  for(left=0, *s++=x, j=NMT; --j;
    *s++ = (x*=69069U) & 0xFFFFFFFFU);
 }

//Used in the function which generates random numbers
uint32 reloadMT(void){
  register uint32 *p0=state, *p2=state+2, *pM=state+MMT, s0, s1;
  register int    j;

  if(left < -1)
    seedMT(4357U);

  left=NMT-1, next=state+1;

  for(s0=state[0], s1=state[1], j=NMT-MMT+1; --j; s0=s1, s1=*p2++)
    *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KMT : 0U);

  for(pM=state, j=MMT; --j; s0=s1, s1=*p2++)
    *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KMT : 0U);

  s1=state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? KMT : 0U);
  s1 ^= (s1 >> 11);
  s1 ^= (s1 <<  7) & 0x9D2C5680U;
  s1 ^= (s1 << 15) & 0xEFC60000U;
  return(s1 ^ (s1 >> 18));
 }

//Generates a random 32-bit integer.
uint32 randomMT(void){
  uint32 y;

  if(--left < 0)
    return(reloadMT());

  y  = *next++;
  y ^= (y >> 11);
  y ^= (y <<  7) & 0x9D2C5680U;
  y ^= (y << 15) & 0xEFC60000U;
  return(y ^ (y >> 18));
 }

//CALCULATE DISTANCE BETWEEN TWO PARTICLES
double dist(double xi, double xj, double yi, double yj, double zi, double zj){
  double x = xi - xj;
  double y = yi - yj;
  double z = zi - zj;

  double d = sqrt(x*x + y*y + z*z);
  return d;
}

//CALCULATE FORCE BETWEEN TWO PARTICLES
double force(double R, double R0, double K){
  double f = -K*(R - R0);
  return f;
}

//CALCULATE FORCE CONSTANT IN ELASTIC NETWORK
//Mode 1, cutoff mode
double forceK(double R, double CUTOFF){
  if ( R <= CUTOFF ){
    return KLIN*SCA;
  }
  else {
    return 0.0;
  }
}

//Mode 2, Kovacs mode
double kovacs(double R){
  return C*pow((RCONST/R),6)*SCA;
}

//Mode 3, Mixed mode
double mixed(double R, int S, double CUTOFF){
  if ( S <= SLIM ){
    return CSEQ/(S*S);
  }
  else if ( R <= CUTOFF ){
    return pow((CCART/R),EX)*SCA;
  }
  else{
    return 0.0;
  }
}

//Force constant of inter-subunit contacts
double forceK_subunit(double R){
  if ( R <= SUBUNIT_CUTOFF ){
    return KLIN*SCA;
  }
  else {
    return 0.0;
  }
}

//GENERATE GAUSSIAN-DISTRIBUTED RANDOM NUMBERS
void gauss_rand(double sigma, double *r1, double *r2, double *r3){
  double w = 2;
  double r1x;
  double r2x;
  double r3x;

  // Box-Muller transform
  while ( w >= 1 ){
    r1x = (2.0*(double)randomMT()/(double)(MAX_RAND)) - 1.0;
    r2x = (2.0*(double)randomMT()/(double)(MAX_RAND)) - 1.0;
    r3x = (2.0*(double)randomMT()/(double)(MAX_RAND)) - 1.0;

    w = r1x*r1x + r2x*r2x + r3x*r3x;
  }
  w = sqrt(-2.0*log(w)/w);

  *r1 = sigma*r1x*w;
  *r2 = sigma*r2x*w;
  *r3 = sigma*r3x*w;
}

//Gaussian-distributed random numbers with 1 variable
void ranvel(double sigma, double *r1){
  double w = 2;
  double r1x;

  // Box-Muller transform
  while ( w >= 1 ){
    r1x = (2*(double)randomMT()/(double)(MAX_RAND)) - 1;

    w = r1x*r1x;
  }
  w = sqrt(-2*log(w)/w);

  *r1 = sigma*r1x*w;
}

//MAIN METHOD
int main(int argc, char *argv[]) {

  //Start clock to measure running time
  double start_time = omp_get_wtime();

  //Take input from terminal
  printf("Read input: %s %s %s %s %s\n",argv[1],argv[2],argv[3],argv[4],argv[5]);

  char start[50];                           //Name of start pdb file (max 50 characters)
  strcpy(start, argv[1]);
  printf("Start pdb file name: %s\n",start);

  char target[50];                          //Name of target pdb file (max 50 characters)
  strcpy(target, argv[2]);
  printf("Target pdb file name: %s\n",target);

  double CUTOFF = strtod(argv[3],NULL);     //Cutoff distance in elastic network
  printf("Cutoff distance: %lf\n",CUTOFF);

  int MODE = atoi(argv[4]);                 //Choice of mode in elastic network model
  printf("Mode number: %d\n",MODE);

  int unbiasedStep = atoi(argv[5]);         //Number of unbiased steps to accept/reject
  printf("Number of unbiased steps: %d\n",unbiasedStep);


  //Read in start file
  //The number of lines equals the number of particles of the system
  FILE *startfile;
  FILE *targetfile;
  int start_lines = 0;
  int target_lines = 0;
  char ch;

  startfile = fopen(start,"r");
  if( startfile ){
    while(( ch = getc(startfile)) != EOF ){
      if( ch == '\n' ){ ++start_lines; }
    }
  }
  else {
    printf("ERROR: Failed to open file %s\n",start);
    exit(0);
  }
  fclose(startfile);

  //Read in target file
  targetfile = fopen(target,"r");
  if( targetfile ){
    while(( ch = getc(targetfile)) != EOF ){
      if( ch == '\n' ){ ++target_lines; }
    }
  }
  else {
    printf("ERROR: Failed to open file %s\n",target);
    exit(0);
  }
  fclose(targetfile);

  //Make sure start and target files contain the same number of particles
  if (start_lines != target_lines){
    printf("ERROR: Number of particles in start conformation %s and target conformation %s does not agree",start,target);
    exit(0);
  }

  int N = start_lines;                         //Number of particles

  printf("Number of particles calculated to %d\n",N);

  //INITIATE VARIABLES
  int acceptStep = 0;                          //Number of accepted simulation steps
  char *aminoa[N];                             //Amino acid type from PDB file
  double bfactor[N];                           //Temperature B factor from PDB file
  char *chain[N];                              //Chain letter from PDB file
  int d=0,i=0,j=0,k=0;                         //Iteration variables
  double fij;                                  //Force between particle i and j
  char filename[50];                           //Name of output file
  double fx[N],fy[N],fz[N];                    //Forces
  double fxt[N],fyt[N],fzt[N];                 //Forces (temporary)
  double K;                                    //Force constant K (spring constant)
  double occupancy[N];                         //Occupancy from PDB file
  double phi;                                  //Angle for spherical coordinates
  double progressR;                            //Progress variable
  double progressStart;                        //Start value of progress variable (progressR)
  double R0;                                   //Sum of initial distances
  double R;                                    //Sum of current distances
  double **R0Array;                            //Initial distances between particles
  double R0temp;                               //Temporary variable for initial distances
  double randx[N],randy[N],randz[N];           //Gaussian-distributed random numbers
  double rcmx=0,rcmy=0,rcmz=0;                 //Sum of particle coordinates (x)
  double Rs;                                   //Distances of starting structure
  double **Rt;                                 //Distances to target structure
  double Rtest;                                //Candidate distance sum
  double rtx[N],rty[N],rtz[N];                 //Particle coordinate of target file
  double rx[N],ry[N],rz[N];                    //Particle coordinates
  double rxbefore[N],rybefore[N],rzbefore[N];  //Previous accepted positions
  int S;                                       //Particle distance in index
  int step = 0;                                //Simulation step counter
  double targetR = 0;                          //Target sum
  double test_time;                            //Time measurement in each iteration
  double theta;                                //Angle for spherical coordinates
  char *type[N];                               //Type from PDB file (2 characters)
  double V1[N],V2[N],V3[N];                    //Absolute value of velocity
  double vx[N],vy[N],vz[N];                    //Particle velocity
  double vxbefore[N],vybefore[N],vzbefore[N];  //Previous accepted velocities

  //Dynamically allocate memory for 2D arrays to use dynamic memory instead of
  //stack (more space)
  Rt = malloc(N*sizeof(double*));
  R0Array = malloc(N*sizeof(double*));


  for (int i = 0; i<N;++i){
    Rt[i] = malloc(N*sizeof(double));
    R0Array[i] = malloc(N*sizeof(double));
    aminoa[i] = malloc(strlen("GLU") + 1);
    type[i] = malloc(strlen("CA") + 1);
    chain[i] = malloc(strlen("A") + 1);
  }

  //Use a random seed for the random number generator
  seedMT(time(NULL));

  //Read start file
  FILE *startFile;
  startFile = fopen(start, "r");

  char line[85];                              //MAGIC! Length of line in standard PDB file
  char line1[85];                             //MAGIC! Length of line in standard PDB file
  char line2[85];                             //MAGIC! Length of line in standard PDB file
  char line3[85];                             //MAGIC! Length of line in standard PDB file
  int s = 0;

  while (fgets(line, sizeof(line), startFile)){
    //Split input line to handle merged occupancies and b factors,
    //or merged chains and residue numbers in some PDB files
    strcpy(line1, line);
    strcpy(line2, line);
    strcpy(line3, line);
    memmove(line1 + 22, line1 + strlen(line1), strlen(line1));
    memmove(line2, line2 + 22, strlen(line2));
    memmove(line2 + 38, line2 +strlen(line2), strlen(line2));
    memmove(line3, line3 + 60, strlen(line3));

    sscanf(line1, "%*s %*d %s %s %s", type[s], aminoa[s], chain[s]);
    sscanf(line2, "%*d %lf %lf %lf %lf", &rx[s], &ry[s], &rz[s], &occupancy[s]);
    sscanf(line3, "%lf %*s", &bfactor[s]);
    ++s;
  }

  fclose(startFile);

  //Read target file
  FILE *targetFile;
  targetFile = fopen(target, "r");

  int t = 0;
  while (fgets(line, sizeof(line), targetFile)){
    sscanf(line, "%*s %*d %*s %*s %*s %*d %lf %lf %lf %*f %*f %*s", &rtx[t], &rty[t], &rtz[t]);
    ++t;
  }

  fclose(targetFile);

  //Create initial conditions
  //Move coordinates so center of mass is in origo
  for (int i = 0; i < N; ++i){
    rcmx += rx[i]/N;
    rcmy += ry[i]/N;
    rcmz += rz[i]/N;
  }

  for (int i = 0; i < N; ++i){
    rx[i] = rx[i] - rcmx;
    ry[i] = ry[i] - rcmy;
    rz[i] = rz[i] - rcmz;
  }

  //Calculate initial distances
  for (int i = 0; i < N; ++i){
    for (int j = (i+1); j < N; ++j){
      R0temp = dist(rx[i],rx[j],ry[i],ry[j],rz[i],rz[j]);
      R0Array[i][j] = R0temp;
    }
  }

  //Generate initial velocities.
  //Velocities are distributed according to a Maxwell-Boltzmann distribution
  #pragma omp parallel for private(i, theta, phi)
  for (int i = 0; i < N; ++i){

    #pragma omp critical (random)
    {
    ranvel(boltzSigma, &V1[i]);

    //Direction of velocity
    theta = PI*((double)randomMT()/(double)(MAX_RAND));
    phi = 2*PI*((double)randomMT()/(double)(MAX_RAND));
    }
    //Assign new velocities to particles
    vx[i] = fabs(V1[i])*sin(theta)*cos(phi)*POW;
    vy[i] = fabs(V1[i])*sin(theta)*sin(phi)*POW;
    vz[i] = fabs(V1[i])*cos(theta)*POW;
  }

  //Copy positions and velocities to temporary arrays
  #pragma omp parallel for private(i)
  for (int i = 0; i < N; ++i){
    rxbefore[i] = rx[i];
    rybefore[i] = ry[i];
    rzbefore[i] = rz[i];
    vxbefore[i] = vx[i];
    vybefore[i] = vy[i];
    vzbefore[i] = vz[i];
  }

  //Calculate target progress variable
  for (int i = 0; i < N; ++i){
    for (int j = (i+1); j < N; ++j){

      //Calculate distances in start and target structures
      Rs = dist(rx[i],rx[j],ry[i],ry[j],rz[i],rz[j]);
      Rt[i][j] = dist(rtx[i],rtx[j],rty[i],rty[j],rtz[i],rtz[j]);
      targetR = targetR + pow((Rs - Rt[i][j]), 2);
    }
  }

  //Create empty start.flag file
  FILE *startflag;
  sprintf(filename,"start.flag");
  startflag = fopen(filename, "w");
  fclose(startflag);

  //Simulation starts here!
  printf("\nSimulation started...\n");

  for (;;){
    ++step;

    progressR = 0.0;

    //Put forces to zero in every time step
    #pragma omp parallel for private(i)
    for (int i = 0; i < N; ++i){
      fx[i] = 0.0;
      fy[i] = 0.0;
      fz[i] = 0.0;
      fxt[i] = 0.0;
      fyt[i] = 0.0;
      fzt[i] = 0.0;

      #pragma omp critical (random)
      {
      gauss_rand(SIGMA, &randx[i], &randy[i], &randz[i]);
      }
    }

    #pragma omp parallel firstprivate(fxt, fyt, fzt) private(i, j, S, K, R0, R, fij)
    {
      #pragma omp for
      for (int i = 0; i < N; ++i){
        for (int j = (i+1); j < N; ++j){

          R0 = R0Array[i][j];
          R = dist(rx[i],rx[j],ry[i],ry[j],rz[i],rz[j]);

          //Check if two particles belong to the same chain and apply correct
          //elastic network model
          if ( strcmp(chain[i],chain[j]) == 0 ){

            if ( MODE == 1 ){
              K = forceK(R,CUTOFF);
            }
            else if ( MODE == 2 ){
              K = kovacs(R);
            }
            else if ( MODE == 3 ){
              S = fabs(i-j);
              K = mixed(R,S,CUTOFF);
            }
            else{
              printf("Invalid choice of mode!\n");
              exit(0);

            }
          }
          //Compute inter-subunit contact between different chains
          else{
            K = forceK_subunit(R);
          }

          fij = force(R, R0, K);

          //Calculate forces
          fxt[i] += fij*(rx[i] - rx[j])/R;
          fyt[i] += fij*(ry[i] - ry[j])/R;
          fzt[i] += fij*(rz[i] - rz[j])/R;
          fxt[j] -= fij*(rx[i] - rx[j])/R;
          fyt[j] -= fij*(ry[i] - ry[j])/R;
          fzt[j] -= fij*(rz[i] - rz[j])/R;
        }
      }

      #pragma omp barrier
      {
      //Add temporary forces calculated in parallel to force array
      for (int k = 0; k < N; ++k){
        fx[k] += fxt[k];
        fy[k] += fyt[k];
        fz[k] += fzt[k];
      }
      }
    }


    #pragma omp parallel
    {
    #pragma omp for private(i)
    for (int i = 0; i < N; ++i){
      //Calculate new positions
      rx[i] = rx[i] + ar*vx[i] + br*fx[i]*INVMASS*POW + cr*randx[i];
      ry[i] = ry[i] + ar*vy[i] + br*fy[i]*INVMASS*POW + cr*randy[i];
      rz[i] = rz[i] + ar*vz[i] + br*fz[i]*INVMASS*POW + cr*randz[i];

      //Calculate new velocities
      vx[i] = vx[i]*av + bv*fx[i]*INVMASS*POW + cv*randx[i];
      vy[i] = vy[i]*av + bv*fy[i]*INVMASS*POW + cv*randy[i];
      vz[i] = vz[i]*av + bv*fz[i]*INVMASS*POW + cv*randz[i];
    }
    #pragma omp barrier
    }

    //Measure wallclock time to check if the time limit was exceeded
    test_time = omp_get_wtime();
    if ( test_time - start_time < TIME_LIMIT ){

      //Only use biasing every # unbiased steps
      if ( step % unbiasedStep == 0 ){

        #pragma omp parallel for private(i, d, Rtest) reduction(+:progressR)
        //Calculate the progress variable that will be tested
        for (int i = 0; i < N; ++i){
          for (int d = (i+1); d < N; ++d){
            Rtest = dist(rx[i],rx[d],ry[i],ry[d],rz[i],rz[d]);
            progressR += (Rtest - Rt[i][d])*(Rtest - Rt[i][d]);
          }
        }

        //Store first value of progressR as progressStart
        if ( step == 1 ){
          progressStart = progressR;
        }

        //Terminate simulation if progress varialbe < 1% of start progress
        if ( progressR < 0.001 * progressStart){
          printf("\nReached >99.9%% of trajectory. Your simulation is done!\n");
          //Create empty end.flag file
          FILE *endflag;
          sprintf(filename,"end.flag");
          endflag = fopen(filename, "w");
          fclose(endflag);

          exit(0);
        }

        //Accept simulation step
        if (progressR < targetR){
          k = 0;
          acceptStep += 1;

          //Generate output
          if ( acceptStep % 500 == 0 ){
            FILE *outfile;
            sprintf(filename,"DIMS_MD%07d.pdb", acceptStep);
            outfile = fopen(filename, "w");
            for (int i = 0; i < N; ++i){

              //Separate chains for proper visualization of multi-chain structures
              if ( i > 0 && strcmp(chain[i],chain[i-1]) != 0){
                k = 0;
              }

              //Print to file
              fprintf(outfile, "%s%7d%4s%5s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f%12s\n", "ATOM", (i+1), type[i], aminoa[i], chain[i], (k+1), rx[i], ry[i], rz[i], occupancy[i], bfactor[i], "C");
              ++k;
            }
            fclose(outfile);
            printf("Wrote file %s\n",filename);
          }

          #pragma omp parallel for private(i)
          for (int i = 0; i < N; ++i){
            rxbefore[i] = rx[i];
            rybefore[i] = ry[i];
            rzbefore[i] = rz[i];
            vxbefore[i] = vx[i];
            vybefore[i] = vy[i];
            vzbefore[i] = vz[i];
          }

          targetR = progressR;
        }

        //Reject simulation step
        else{
          #pragma omp parallel for private(i, theta, phi)
          for (int i = 0; i < N; ++i){
            rx[i] = rxbefore[i];
            ry[i] = rybefore[i];
            rz[i] = rzbefore[i];

            //Generate new Maxwell-Boltzmann distributed velocities
            #pragma omp critical (random)
            {
            ranvel(boltzSigma, &V1[i]);

            //Direction of velocity
            theta = PI*((double)randomMT()/(double)(MAX_RAND));
            phi = 2*PI*((double)randomMT()/(double)(MAX_RAND));
            }
            //Assign new velocities to particles
            vx[i] = fabs(V1[i])*sin(theta)*cos(phi)*POW;
            vy[i] = fabs(V1[i])*sin(theta)*sin(phi)*POW;
            vz[i] = fabs(V1[i])*cos(theta)*POW;

          }
        }
      }
    }

    //Terminate simulation if the time limit was exceeded
    else {
      printf("\nSimulation terminated due to exceeded time limit\n");

      //Create checkpoint file before termination (this is not neccesarily an
      //accepted simulation step!)
      FILE *cptfile;
      sprintf(filename,"DIMS_MD%07d_cpt.pdb", acceptStep);
      cptfile = fopen(filename, "w");
      for (int i = 0; i < N; ++i){

        //Separate chains for proper visualization of multi-chain structures
        if ( i > 0 && strcmp(chain[i],chain[i-1]) != 0){
          k = 0;
        }

        //Print to file
        fprintf(cptfile, "%s%7d%4s%5s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f%12s\n", "ATOM", (i+1), type[i], aminoa[i], chain[i], (k+1), rx[i], ry[i], rz[i], occupancy[i], bfactor[i], "C");
        ++k;
      }
      fclose(cptfile);
      printf("Wrote checkpoint file: %s\n",filename);

      //Create empty end.flag file
      FILE *endflag;
      sprintf(filename,"end.flag");
      endflag = fopen(filename, "w");
      fclose(endflag);

      exit(0);
    }
  }

  //Create empty end.flag file
  FILE *endflag;
  sprintf(filename,"end.flag");
  endflag = fopen(filename, "w");
  fclose(endflag);

  printf("Simulation finished!\n");

  free(Rt);
  free(R0Array);

  //Stop clock to measure running time
  //double end_time = omp_get_wtime();
  //double seconds = ( end_time - start_time );
  //printf("Simulation took %f seconds for %d time steps\n", seconds, NSTEPS);

  return 0;
}
