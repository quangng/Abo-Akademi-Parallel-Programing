/*
 * Parallel programming 2014
 * Assignment. Computing the two-point angular correlation function 
 * Author: Vu Nguyen 
 * Abo student number: 71141
 * Date: 1 October 2014
 * Compile with mpicc -O3 galaxyz_p.c -o galaxyz_p
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define binsperdegree 4		// Nr of bins per degree
#define totaldegrees 64		// Nr of degrees

/* Count how many lines the input file has */
int count_lines (FILE *infile) {
	char readline[80];      /* Buffer for file input */
	int lines=0;
	while( fgets(readline,80,infile) != NULL ) lines++;
	rewind(infile);  /* Reset the file to the beginning */
	return(lines);
}

/* Read input data from the file, convert to cartesian coordinates 
   and write them to arrays x, y and z */
void read_data(FILE *infile, int n, float *x, float *y, float *z) {
	char readline[80];      /* Buffer for file input */
	float ra, dec, theta, phi, dpi;
	int i=0;
	dpi = acos(-1.0);
	while( fgets(readline,80,infile) != NULL )  /* Read a line */
    {
		sscanf(readline,"%f %f",&ra, &dec);  /* Read a coordinate pair */
		/* Convert to cartesian coordinates */
		phi   = ra * dpi/180.0;
		theta = (90.0-dec)*dpi/180;
		x[i] = sinf(theta)*cosf(phi);
		y[i] = sinf(theta)*sinf(phi);
		z[i] = cosf(theta);
		++i;
    } 
	fclose(infile);
}

/* Compute the angle between two observations p and q and 
   add it to the histogram */
void add_histogram (float px, float py, float pz, 
					float qx, float qy, float qz, long int *histogram, 
					const float pi, const float costotaldegrees) {
	float theta;
    float degreefactor = 180.0/pi*binsperdegree;
    int bin;
	theta = px*qx + py*qy + pz*qz;
	if ( theta >= costotaldegrees ) {   /* Skip if theta < costotaldegrees */
		if ( theta > 1.0 ) theta = 1.0;
		/* Calculate which bin to increment */
		/* histogram [(int)(acos(theta)*180.0/pi*binsperdegree)] += 1L; */
		bin = (int)(acosf(theta)*degreefactor); 
		histogram[bin]++;
	}
}

/* Find the min between two numbers */
int min (int a, int b) {
	int min;
	
	if (a < b) min = a;
	else min = b;
	
	return min;
}


/* Main program */
int main(int argc, char* argv[]) { 
	int rank,						// Process id
		size, 						// Number of processes
		err,						// error return value
		first_index,				// Start index assigned to a process in data composition
		last_index, 				// End index assigned to a process in data composition
		center_index,				// Center index of an array in data composition
		nmin, 						// Number of elements per process
		nleft, 						// Number of elements left over
		first, 						// Index of first element handled by this process
		M; 							// Number of elements handled by this process


	double startTime, endTime;	
  	int Nooflines_Real;  			// Nr of lines in real data
  	int Nooflines_Sim;   			// Nr of lines in random data 
	float *xd_real, *yd_real, *zd_real;								// Arrays for real data
  	float *xd_sim , *yd_sim , *zd_sim; 								// Arrays for random data
	long int *histogramDD, *histogramDR, *histogramRR; 				// Arrays for histograms 
	long int *histogramDD_tmp, *histogramDR_tmp, *histogramRR_tmp;	// Temporary arrays for histograms
	FILE *infile_real, *infile_sim, *outfile; 
	int nr_of_bins = binsperdegree * totaldegrees;					// Total number of intervals/bins
	int i, j;										
	float pi, costotaldegrees;										// Value of PI and cosine of totaldegrees
	long int TotalCountDD, TotalCountRR, TotalCountDR;
	double NSimdivNReal, w;											

	
	/* Initialize MPI */
  	err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS) {
		printf("MPI_Init failed!\n");
		exit(1);
	}
	
	startTime = MPI_Wtime();	
	/* Calculate value of PI and cosine of totaldegrees */
	pi = acosf(-1.0);
  	costotaldegrees = (float)(cos(totaldegrees/180.0*pi));
	
  	err = MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (err != MPI_SUCCESS) {
		printf("MPI_Comm_size failed!\n");
		exit(1);
	}
	
  	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (err != MPI_SUCCESS) {
		printf("MPI_Comm_rank failed!\n");
		exit(1);
	}

	/* Check that we run on at least two processors */
	if (size < 2) {
		printf("You have to use at least two processors to run this program!\n");
		err = MPI_Finalize();
		if (err != MPI_SUCCESS) {
			printf("MPI_Finalize failed!\n");
			exit(1);
		}
	}
	
	/* Read real data file and random data file */
	infile_real = fopen(argv[1],"r");
  	if ( infile_real == NULL ) {
    	printf("Unable to open %s\n",argv[1]);
   		return(0);
  	}
	
	infile_sim = fopen(argv[2],"r");
  	if ( infile_sim == NULL ) {
    	printf("Unable to open %s\n",argv[2]);
    	return(0);
 	}

	/* Process 0 count the number of lines of real data file and of random data file 
	 * Then send these numbers to all other processes. Other processes receive these numbers
	 * to allocate arrays for x, y, and z values
	 */
	if (rank == 0) { 
		Nooflines_Real = count_lines(infile_real);
		Nooflines_Sim = count_lines(infile_sim);	
		err = MPI_Bcast(&Nooflines_Real, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS) {
			printf("MPI_Bcast failed!\n");
			exit(1);
		}
		
		err = MPI_Bcast(&Nooflines_Sim, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS) {
			printf("MPI_Bcast failed!\n");
			exit(1);
		}
	} else { 
		err = MPI_Bcast(&Nooflines_Real, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS) {
			printf("MPI_Bcast failed!\n");
			exit(1);
		}

		err = MPI_Bcast(&Nooflines_Sim, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS) {
			printf("MPI_Bcast failed!\n");
			exit(1);
		}
	}
	
	/* Allocate arrays for x, y and z values */
  	xd_real = (float *)calloc(Nooflines_Real, sizeof(float));
	if (xd_real == NULL) {
		printf("calloc failed\n");
		exit(1);
	}

  	yd_real = (float *)calloc(Nooflines_Real, sizeof(float));
	if (yd_real == NULL) {
		printf("calloc failed\n");
		exit(1);
	}

  	zd_real = (float *)calloc(Nooflines_Real, sizeof(float));
	if (zd_real == NULL) {
		printf("calloc failed\n");
		exit(1);
	}

	xd_sim = (float *)calloc(Nooflines_Sim, sizeof(float));
	if (xd_sim == NULL) {
		printf("calloc failed\n");
		exit(1);
	}
	
  	yd_sim = (float *)calloc(Nooflines_Sim, sizeof(float)); 
	if (yd_sim == NULL) {
		printf("calloc failed\n");
		exit(1);
	}
	
  	zd_sim = (float *)calloc(Nooflines_Sim, sizeof(float)); 
	if (zd_sim == NULL) {
		printf("calloc failed\n");
		exit(1);
	}

	/* Read the file with real input data */
  	read_data(infile_real, Nooflines_Real, xd_real, yd_real, zd_real);
	read_data(infile_sim, Nooflines_Sim, xd_sim, yd_sim, zd_sim);	
	
	/* Allocate arrays for the histograms DD, DR, RR */
	histogramDD = (long int *)calloc(nr_of_bins+1, sizeof(long int));
	if (histogramDD == NULL) {
		printf("calloc failed\n");
		exit(1);
	} 

  	histogramDR = (long int *)calloc(nr_of_bins+1, sizeof(long int));
	if (histogramDR == NULL) {
		printf("calloc failed\n");
		exit(1);
	}
	
  	histogramRR = (long int *)calloc(nr_of_bins+1, sizeof(long int));
	if (histogramRR == NULL) {
		printf("calloc failed\n");
		exit(1);
	}
	
	histogramDD_tmp = (long int *)calloc(nr_of_bins+1, sizeof(long int)); 
	if (histogramDD_tmp == NULL) {
		printf("calloc failed\n");
		exit(1);
	}
	
	histogramDR_tmp = (long int *)calloc(nr_of_bins+1, sizeof(long int));
	if (histogramDR_tmp == NULL) {
		printf("calloc failed\n");
		exit(1);
	}	

	histogramRR_tmp = (long int *)calloc(nr_of_bins+1, sizeof(long int));
	if (histogramRR_tmp == NULL) {
		printf("calloc failed\n");
		exit(1);
	}

	
	/* Apply domain decomposition and calculate histograms DD */
	center_index = (int)(Nooflines_Real-1)/2;
	for (i = rank; i < center_index; i += size) {
		first_index = i;
		last_index = Nooflines_Real - 2 - i;
		for (j = first_index+1; j < Nooflines_Real; j++)
			add_histogram (xd_real[first_index], yd_real[first_index], zd_real[first_index], 
	        			   xd_real[j], yd_real[j], zd_real[j], histogramDD, pi, costotaldegrees);
		
		for (j = last_index+1; j < Nooflines_Real; j++)
			add_histogram (xd_real[last_index], yd_real[last_index], zd_real[last_index], 
	        			   xd_real[j], yd_real[j], zd_real[j], histogramDD, pi, costotaldegrees);
	}
	
	if ((Nooflines_Real-1)%2 == 1) {
		if (rank == 0) {	// If there is a center index, process 0 will take care of the calculation
			for (j = center_index+1; j < Nooflines_Real; j++)
				add_histogram (xd_real[center_index], yd_real[center_index], zd_real[center_index], 
							   xd_real[j], yd_real[j], zd_real[j], histogramDD, pi, costotaldegrees);
		}
	}

	for (i = 0; i <= nr_of_bins; ++i) 
		histogramDD[i] *= 2L;
	if (rank == 0)	// process 0 calculates the number of (i, i) pairs
		histogramDD[0] += ((long)(Nooflines_Real));

	/* Apply domain decomposition and calculate histograms RR */
	center_index = (int)(Nooflines_Sim-1)/2;
	for (i = rank; i < center_index; i += size) {
		first_index = i;
		last_index = Nooflines_Sim - 2 - i;
		for (j = first_index+1; j < Nooflines_Sim; j++)
			add_histogram (xd_sim[first_index], yd_sim[first_index], zd_sim[first_index], 
	        			   xd_sim[j], yd_sim[j], zd_sim[j], histogramRR, pi, costotaldegrees);
						   
		for (j = last_index+1; j < Nooflines_Sim; j++)
			add_histogram (xd_sim[last_index], yd_sim[last_index], zd_sim[last_index], 
	        			   xd_sim[j], yd_sim[j], zd_sim[j], histogramRR, pi, costotaldegrees);
	}

	if ((Nooflines_Sim-1)%2 == 1) { 
		if (rank == 0) {
			for (j = center_index+1; j < Nooflines_Sim; j++) 
				add_histogram (xd_sim[center_index], yd_sim[center_index], zd_sim[center_index], 
	        			   	   xd_sim[j], yd_sim[j], zd_sim[j], histogramRR, pi, costotaldegrees);
		}	
	}

	for (i = 0; i <= nr_of_bins; i++) 
		histogramRR[i] *= 2L;
	if (rank == 0) 
		histogramRR[0] += ((long)(Nooflines_Sim));
		
	
	/* Apply domain decomposition and calculate histogram DR */
	nmin = Nooflines_Real/size;		
	nleft = Nooflines_Real%size;	
	
	if (rank < nleft) M = nmin + 1;
	else M = nmin;
	
	first = rank * nmin + min(rank, nleft);
	for (i = first; i < first+M; i++) {
		for (j = 0; j < Nooflines_Sim; j++)
			add_histogram (xd_real[i], yd_real[i], zd_real[i], 
						   xd_sim[j], yd_sim[j], zd_sim[j], histogramDR, pi, costotaldegrees);
	}
	
	/* Process 0 sum up arrays of histograms DD, DR, RR from other processes */
	err = MPI_Reduce(histogramDD, histogramDD_tmp, nr_of_bins+1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (err != MPI_SUCCESS) {
		printf("MPI_Reduce failed!\n");
		exit(1);
	}
	err = MPI_Reduce(histogramDR, histogramDR_tmp, nr_of_bins+1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (err != MPI_SUCCESS) {
		printf("MPI_Reduce failed!\n");
		exit(1);
	}
	
	err = MPI_Reduce(histogramRR, histogramRR_tmp, nr_of_bins+1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (err != MPI_SUCCESS) {
		printf("MPI_Reduce failed!\n");
		exit(1);
	}
	
			
	if (rank == 0) {
		/* Count the total number of values in DD, DR, and RR histograms */
		TotalCountDD = 0L;
		TotalCountDR = 0L;
		TotalCountRR = 0L;
  		for (i = 0; i <= nr_of_bins; ++i) {
			TotalCountDD += (long)(histogramDD_tmp[i]); 
			TotalCountDR += (long)(histogramDR_tmp[i]);
			TotalCountRR += (long)(histogramRR_tmp[i]);
		}
    		
  		printf("DD histogram count = %ld\n", TotalCountDD);
		printf("DR histogram count = %ld\n", TotalCountDR);
		printf("RR histogram count = %ld\n\n", TotalCountRR);
		
		/* Process 0 write the histograms both to display and outfile */
  		outfile = fopen(argv[3],"w");	// Open the output file
  		if ( outfile == NULL ) {
    		printf("Unable to open %s\n",argv[3]);
    		return(-1);
  		}

		NSimdivNReal = ((double)(Nooflines_Sim))/((double)(Nooflines_Real));
		printf("bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
		fprintf(outfile,"bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
		for ( i = 0; i < nr_of_bins; ++i ) {
			w = 1.0 + NSimdivNReal*NSimdivNReal*histogramDD_tmp[i]/histogramRR_tmp[i] 
				-2.0*NSimdivNReal*histogramDR_tmp[i]/((double)(histogramRR_tmp[i]));
			printf(" %6.3f      %3.6f\t%15ld\t%15ld\t%15ld\n",((float)i+0.5)/binsperdegree, w, 
	     			histogramDD[i], histogramDR[i], histogramRR[i]);
      		fprintf(outfile,"%6.3f\t%15lf\t%15ld\t%15ld\t%15ld\n",((float)i+0.5)/binsperdegree, w, 
	      			histogramDD[i], histogramDR[i], histogramRR[i]);
    	} 

		fclose(outfile);
	}
	
	/* Free all allocated arrays */
  	free(histogramDD); free(histogramDD_tmp);
  	free(histogramDR); free(histogramDR_tmp);
  	free(histogramRR); free(histogramRR_tmp);
  	free(xd_sim); free(yd_sim); free(zd_sim);
  	free(xd_real); free(yd_real); free(zd_real);

	endTime = MPI_Wtime();
	if (rank == 0) 
		printf("Running parallel program took %3.1lf seconds\n", endTime-startTime);
	MPI_Finalize();	

	return 0;
}
