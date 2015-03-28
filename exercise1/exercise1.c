/*
 * Parallel programming 2014
 * Exercise 1. Communication in a ring of processes
 * Author: Vu Nguyen 
 * Student number: 71141
 * University of Turku
 * Date: 1 October 2014

 * Compile with
 * mpicc -O3 -std=c99 model.c -o model
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define K 1024            /* One Kilobyte */
#define M 1000*K          /* One Megabyte */
#define MAXSIZE 1000*M    /* 1000 Megabytes = 1 GB */

int main(int argc, char* argv[]) {
	const int msgtag = 42;	/* Arbitrarily choose message tag to be 42 */
	const int sizetag = 41;	/* Arbitrarly choose size tag to be 41 */
	int rank, size, err;
	int messageSize, inmessageSize;
  	char *outbuffer, *inbuffer;
	int prev, next;
  	double startTime, endTime;

  	/* Initialize MPI */
  	err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS) {
		printf("MPI_Init failed!\n");
		exit(1);
	}

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
	
	/* Find out the id number of the previous and the next process */
	prev = (rank - 1 + size) % size;
	next = (rank + 1) % size;

	/* Process 0 in the ring will do this */
  	if (rank == 0) {
    	printf("Number of processors is %d\n", size);

		do {
			printf("Please give an input size in bytes: \n");
    		fflush(NULL);
    		scanf("%d", &messageSize);
				
			/* Boundary checking: If the message size > MAXSIZE or message size <= 0, 
			 * then send the size of the message to the rest of the processes in the ring and terminate
			 * process 0 after that.
			 */
			if (messageSize > MAXSIZE) {
      				printf("Sorry, that's too much memory!\n\n");
      				messageSize = 0;
			} 

			if (messageSize <  0) {
				printf("Input size cannot be negative!\n");
				messageSize = 0;
			} 
			
			if (messageSize == 0) {
				err = MPI_Send(&messageSize, 1, MPI_INT, next, sizetag, MPI_COMM_WORLD);
				if (err != MPI_SUCCESS) {
					printf("MPI_Send failed!\n");
					exit(1);
				}

				err = MPI_Recv(&inmessageSize, 1, MPI_INT, prev, sizetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (err != MPI_SUCCESS) {
					printf("MPI_Recv failed!\n");
					exit(1);
				}

				break;
			} else {
				/* Send the size of the message to other processes and wait for it from the last process */
				err = MPI_Send(&messageSize, 1, MPI_INT, next, sizetag, MPI_COMM_WORLD);
				if (err != MPI_SUCCESS) {
					printf("MPI_Send failed!\n");
					exit(1);
				}

				err = MPI_Recv(&inmessageSize, 1, MPI_INT, prev, sizetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (err != MPI_SUCCESS) {
					printf("MPI_Recv failed!\n");
					exit(1);
				}
				
				/* Dynamically allocate memory for the sending and receiving buffer, and 
				 * generate a message of size messageSize
				 */
				outbuffer = (char *)malloc(messageSize * sizeof(char));
				inbuffer = (char *)malloc(messageSize * sizeof(char));
				memset(outbuffer, 1, messageSize);
				
				/* Send a message of size messageSize around the ring and measure travel time */
				startTime = MPI_Wtime();
				err = MPI_Send(outbuffer, messageSize, MPI_CHAR, next, msgtag, MPI_COMM_WORLD);
				if (err != MPI_SUCCESS) {
					printf("MPI_Send failed!\n");
					exit(1);
				}

				err = MPI_Recv(inbuffer, messageSize, MPI_CHAR, prev, msgtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (err != MPI_SUCCESS) {
					printf("MPI_Recv failed!\n");
					exit(1);
				}
				endTime = MPI_Wtime();
				printf("Sending message around the ring took: %f s\n", endTime - startTime);
				
				/* Deallocate memory from the heap process 0*/
				free(inbuffer);
				free(outbuffer);	
			}			
		} while (messageSize != 0);
    		
  		err = MPI_Finalize();
		if (err != MPI_SUCCESS) {
			printf("Error in MPI_Finalize!\n");
			exit(1);
		} 
		exit(0);
  	} 

	/* Other processes in the ring will do this */
	 else {

		do {
			/* Receive the size of the message from the previous process and send forward it to the next process */
			err = MPI_Recv(&messageSize, 1, MPI_INT, prev, sizetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (err != MPI_SUCCESS) {
				printf("MPI_Recv failed!\n");
				exit(1);
			}

			err = MPI_Send(&messageSize, 1, MPI_INT, next, sizetag, MPI_COMM_WORLD);
			if (err != MPI_SUCCESS) {
				printf("MPI_Send failed!\n");
				exit(1);
			}
			
			if (messageSize == 0)
				break;
			else {
				/* Size of the message is valid -> allocate memory to receive message of size messageSize*/
				inbuffer = (char *)malloc(messageSize * sizeof (char));

				/* Receive the message of size messageSize from the previous process and forward it to the next process */
				err = MPI_Recv(inbuffer, messageSize, MPI_CHAR, prev, msgtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (err != MPI_SUCCESS) {
					printf("MPI_Recv failed!\n");
					exit(1);
				}

				err = MPI_Send(inbuffer, messageSize, MPI_CHAR, next, msgtag, MPI_COMM_WORLD);
				if (err != MPI_SUCCESS) {
					printf("MPI_Send failed!\n");
					exit(1);
				}
				
				/* Deallocate memory from the heap */
				free(inbuffer);	
			}
		} while (messageSize != 0);

		err = MPI_Finalize();
		if (err != MPI_SUCCESS) {
			printf("MPI_Finalize failed!\n");
			exit(1);
		}
		exit(0);
	}  
}	/* End of main function*/


