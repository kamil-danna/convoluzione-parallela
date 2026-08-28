import matplotlib.pyplot as plt
import numpy as np

# Impostazioni di stile generali
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ---------------------------------------------------------
# 1. Grafico Strong Scaling (Speedup)
# ---------------------------------------------------------
cores = [1, 2, 4]
speedup_omp = [0.99, 1.90, 3.48]
speedup_mpi = [1.00, 1.95, 3.77]
ideal = [1, 2, 4]

plt.figure(figsize=(8, 5))
plt.plot(cores, speedup_omp, marker='o', linewidth=2, label='OpenMP', color=colors[0])
plt.plot(cores, speedup_mpi, marker='s', linewidth=2, label='MPI', color=colors[1])
plt.plot(cores, ideal, linestyle='--', color='gray', label='Ideale')

plt.title('Strong Scaling: Speedup (Kernel 5x5)', fontsize=14)
plt.xlabel('Numero di Core / Processi', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.xticks(cores)
plt.legend()
plt.savefig('1_Speedup_4096.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 2. Grafico Efficienza Parallela
# ---------------------------------------------------------
eff_omp = [0.99, 0.95, 0.87]
eff_mpi = [1.00, 0.98, 0.94]

plt.figure(figsize=(8, 5))
plt.plot(cores, eff_omp, marker='o', linewidth=2, label='OpenMP', color=colors[0])
plt.plot(cores, eff_mpi, marker='s', linewidth=2, label='MPI', color=colors[1])

plt.title('Efficienza Parallela (Kernel 5x5)', fontsize=14)
plt.xlabel('Numero di Core / Processi', fontsize=12)
plt.ylabel('Efficienza', fontsize=12)
plt.xticks(cores)
plt.ylim(0, 1.2)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5) # Linea del 100%
plt.legend()
plt.savefig('2_Efficienza_4096.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. Istogramma Configurazioni Ibride
# ---------------------------------------------------------
labels_ibrido = ['4 MPI x 1 OMP', '2 MPI x 2 OMP', '1 MPI x 4 OMP']
eff_ibrido = [0.92, 0.93, 0.59]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels_ibrido, eff_ibrido, color=colors[0], width=0.5)

plt.title('Efficienza Configurazioni Ibride', fontsize=14)
plt.ylabel('Efficienza', fontsize=12)
plt.ylim(0, 1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontsize=10)

plt.savefig('3_Ibrido_4096.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 4. Istogramma Intensità Computazionale (Kernel)
# ---------------------------------------------------------
kernels = ['3x3', '5x5', '7x7', '9x9']
eff_kernel = [0.80, 0.87, 0.93, 0.97]

plt.figure(figsize=(8, 5))
bars2 = plt.bar(kernels, eff_kernel, color=colors[1], width=0.5)

plt.title('Efficienza OpenMP (4 Thread) al variare del Kernel', fontsize=14)
plt.xlabel('Dimensione Kernel', fontsize=12)
plt.ylabel('Efficienza', fontsize=12)
plt.ylim(0, 1.2)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontsize=10)

plt.savefig('4_Kernel_4096.png', dpi=300, bbox_inches='tight')
plt.close()