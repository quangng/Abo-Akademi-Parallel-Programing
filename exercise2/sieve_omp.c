/*
Parallel programming 2014
Exercise 2. Implementation of prime number sieve using  OpenMP
Student name: Vu Nguyen (71141)
Compiled with gcc -O3 sieve_omp.c -o sieve_omp -fopenmp -lm
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const char unmarked = (char)0;
const char marked = (char)1;

int main(int argc, char *argv[]) {
	unsigned long int i;	
	unsigned long int k;
	unsigned long int N;	//value of N
	char *prime;			//a byte array of size N+1
	unsigned long int primecount = 0;
	unsigned long int maxprime = 0;
	unsigned long int max = 0;
	double startTime, endTime;
	
	/* Check that the user enters the command correctly */
	if (argc != 2) {
		printf("Usage: %s N\n", argv[0]);
		printf("Usage: N is any positive integer number larger than 2\n");
		exit(1);
	}
	
	/* Start measuring time */
	startTime = omp_get_wtime();
	
	/* Allocate an array of size N+1 */
	N = atoi(argv[1]);
	prime = (char *) malloc((N+1) * sizeof(char));
	if (prime == NULL) {
		printf("malloc failed!");
		exit(1);
	}
	
	/* Fork a team of threads */
	#pragma omp parallel private(i, k) shared(prime, maxprime) firstprivate(max)
	{
		/* Initialize the prime array to unmarked */
		#pragma omp for
		for (i = 0; i < N+1; i++)
			prime[i] = unmarked;
			
		/* Find out composite numbers and mark it */
		#pragma omp for schedule(dynamic)
		for (i = 2; i <= (int)sqrt(N); i++) {
			if(prime[i] == unmarked) {
				for (k = i*i; k <= N; k += i)
					prime[k] = marked;
			}
		}
		
		/* Calculate the number of prime numbers and the largest of those*/
		#pragma omp for
		for (i = 2; i < N+1; i++) {
			if (prime[i] == unmarked) {
				#pragma omp atomic
				primecount++;
				
				
				if (i > max) max = i;
				#pragma omp critical 
				{
					if(max > maxprime) maxprime = max;
				}
			}	
		}
	}	// End of parallel region
	
	printf("The number of primes: %ld\n", primecount);
	printf("The largest prime: %ld\n", maxprime);
	
	endTime = omp_get_wtime(); 
	printf("%.6f ms\n", (endTime - startTime)*1e3);
	
	free(prime);
	return 0;
}