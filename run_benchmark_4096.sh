#!/bin/bash

IMG="input_4096.pgm"
OUT="output_4096.pgm"
RUNS=5
CSV="benchmark_results_4096.csv"

echo "Test,Programma,Processi_MPI,Thread_OMP,Kernel,Tempo_Medio_Sec" > $CSV

run_test() {
    TEST_NAME=$1
    PROG=$2
    MPI_NP=$3
    OMP_TH=$4
    K=$5

    export OMP_NUM_THREADS=$OMP_TH
    echo -n "--> $TEST_NAME [$PROG | MPI:$MPI_NP | OMP:$OMP_TH | Ker:$K]... "

    if [ "$PROG" == "./convolution" ]; then
        AVG=$(for i in $(seq 1 $RUNS); do ./convolution $IMG $OUT $K; done | grep "Tempo" | awk '{sum+=$4} END {print sum/NR}')
    elif [ "$PROG" == "./conv_omp" ]; then
        AVG=$(for i in $(seq 1 $RUNS); do ./conv_omp $IMG $OUT $K; done | grep "Tempo" | awk '{sum+=$4} END {print sum/NR}')
    elif [ "$PROG" == "./conv_mpi" ]; then
        AVG=$(for i in $(seq 1 $RUNS); do mpirun --allow-run-as-root -np $MPI_NP ./conv_mpi $IMG $OUT $K; done | grep "Tempo" | awk '{sum+=$7} END {print sum/NR}')
    elif [ "$PROG" == "./conv_hybrid" ]; then
        AVG=$(for i in $(seq 1 $RUNS); do mpirun --allow-run-as-root -np $MPI_NP ./conv_hybrid $IMG $OUT $K; done | grep "Tempo" | awk '{sum+=$12} END {print sum/NR}')
    fi

    echo "$TEST_NAME,$PROG,$MPI_NP,$OMP_TH,$K,$AVG" >> $CSV
    echo "Media: $AVG sec"
}

echo "Inizio Benchmark su immagine 4096x4096..."
echo "========================================================================"

echo "--- 1. TEST DI STRONG SCALING (Kernel fisso a 5) ---"
run_test "Baseline Sequenziale" "./convolution" 1 1 5
run_test "OpenMP 1 Thread" "./conv_omp" 1 1 5
run_test "OpenMP 2 Thread" "./conv_omp" 1 2 5
run_test "OpenMP 4 Thread" "./conv_omp" 1 4 5
run_test "MPI 1 Processo" "./conv_mpi" 1 1 5
run_test "MPI 2 Processi" "./conv_mpi" 2 1 5
run_test "MPI 4 Processi" "./conv_mpi" 4 1 5

echo ""
echo "--- 2. CONFRONTO IBRIDO (Totale 4 Lavoratori, Kernel fisso a 5) ---"
run_test "Ibrido 4 MPI x 1 OMP" "./conv_hybrid" 4 1 5
run_test "Ibrido 2 MPI x 2 OMP" "./conv_hybrid" 2 2 5
run_test "Ibrido 1 MPI x 4 OMP" "./conv_hybrid" 1 4 5

echo ""
echo "--- 3. INTENSITA' COMPUTAZIONALE (Variazione Kernel) ---"
for k in 3 5 7 9; do
    run_test "Baseline Sequenziale K=$k" "./convolution" 1 1 $k
    run_test "OpenMP 4 Thread K=$k" "./conv_omp" 1 4 $k
done

echo "========================================================================"
echo "Test Completati! I risultati sono salvati in $CSV"
