#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -- Strutture e utilità base --
typedef struct
{
    int width, height, max_val;
    unsigned char *data;
} PGMImage;

int clamp(int val, int min, int max)
{
    if (val < min)
        return min;
    if (val >= max)
        return max - 1;
    return val;
}

PGMImage *read_pgm(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Errore apertura %s\n", filename);
        exit(1);
    }
    PGMImage *img = (PGMImage *)malloc(sizeof(PGMImage));
    char format[3];
    o if (fscanf(file, "%2s", format) != 1)
    {
        fprintf(stderr, "Errore: impossibile leggere il formato del file.\n");
        exit(1);
    }

    if (strcmp(format, "P5") != 0)
    {
        fprintf(stderr, "Errore: il file non è un PGM binario (P5)\n");
        exit(1);
    }

    int c = getc(file);
    while (c == '#' || c == '\n' || c == ' ' || c == '\r')
    {
        if (c == '#')
            while (getc(file) != '\n')
                ;
        else
            c = getc(file);
    }
    ungetc(c, file);

    if (fscanf(file, "%d %d %d", &img->width, &img->height, &img->max_val) != 3)
    {
        fprintf(stderr, "Errore: impossibile leggere le dimensioni o il valore massimo.\n");
        exit(1);
    }
    fgetc(file);

    int size = img->width * img->height;
    img->data = (unsigned char *)malloc(size);

    if (fread(img->data, 1, size, file) != (size_t)size)
    {
        fprintf(stderr, "Errore: lettura dei pixel fallita o file troncato.\n");
        exit(1);
    }

    fclose(file);
    return img;
}

void write_pgm(const char *filename, PGMImage *img)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Errore: impossibile creare %s\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img->width, img->height, img->max_val);

    int size = img->width * img->height;

    if (fwrite(img->data, 1, size, file) != (size_t)size)
    {
        fprintf(stderr, "Errore: scrittura dei pixel fallita.\n");
        exit(1);
    }

    fclose(file);
}

float *generate_gaussian_kernel(int k_size)
{
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));
    float sum = 0.0f, s = 2.0f;
    int r = k_size / 2;
    for (int x = -r; x <= r; x++)
    {
        for (int y = -r; y <= r; y++)
        {
            float val = (exp(-(x * x + y * y) / s)) / (M_PI * s);
            kernel[(x + r) * k_size + (y + r)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < k_size * k_size; i++)
        kernel[i] /= sum;
    return kernel;
}

int main(int argc, char *argv[])
{
    // Inizializziamo MPI con il supporto per i thread
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        printf("Attenzione: Il supporto ai thread MPI non è sufficiente!\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4)
    {
        if (rank == 0)
            printf("Uso: %s <input.pgm> <output.pgm> <k_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int k_size = atoi(argv[3]);
    int r = k_size / 2;

    PGMImage *img = NULL;
    int width = 0, height = 0, max_val = 0;

    if (rank == 0)
    {
        img = read_pgm(input_file);
        width = img->width;
        height = img->height;
        max_val = img->max_val;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *kernel = generate_gaussian_kernel(k_size);

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows = height / size + (i < (height % size) ? 1 : 0);
            sendcounts[i] = rows * width;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    int local_rows = height / size + (rank < (height % size) ? 1 : 0);
    unsigned char *local_data = malloc(local_rows * width);
    unsigned char *local_out = malloc(local_rows * width);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Distribuzione
    MPI_Scatterv(rank == 0 ? img->data : NULL, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 local_data, local_rows * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Scambio Halo
    int padded_rows = local_rows + 2 * r;
    unsigned char *padded_data = malloc(padded_rows * width);
    memcpy(padded_data + (r * width), local_data, local_rows * width);

    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Sendrecv(padded_data + (r * width), r * width, MPI_UNSIGNED_CHAR, top_neighbor, 0,
                 padded_data, r * width, MPI_UNSIGNED_CHAR, top_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(padded_data + (local_rows * width), r * width, MPI_UNSIGNED_CHAR, bottom_neighbor, 1,
                 padded_data + ((r + local_rows) * width), r * width, MPI_UNSIGNED_CHAR, bottom_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == 0)
    {
        for (int i = 0; i < r; i++)
            memcpy(padded_data + (i * width), padded_data + (r * width), width);
    }
    if (rank == size - 1)
    {
        for (int i = 0; i < r; i++)
            memcpy(padded_data + ((r + local_rows + i) * width), padded_data + ((r + local_rows - 1) * width), width);
    }

// Aggiungiamo OpenMP per calcolare il nostro blocco locale!
#pragma omp parallel for
    for (int y = 0; y < local_rows; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ky++)
            {
                for (int kx = -r; kx <= r; kx++)
                {
                    int px = clamp(x + kx, 0, width);
                    int py = y + r + ky;
                    float weight = kernel[(ky + r) * k_size + (kx + r)];
                    sum += padded_data[py * width + px] * weight;
                }
            }
            int final_pixel = (int)sum;
            if (final_pixel < 0)
                final_pixel = 0;
            if (final_pixel > 255)
                final_pixel = 255;
            local_out[y * width + x] = (unsigned char)final_pixel;
        }
    }

    // Raccolta
    PGMImage *out_img = NULL;
    if (rank == 0)
    {
        out_img = (PGMImage *)malloc(sizeof(PGMImage));
        out_img->width = width;
        out_img->height = height;
        out_img->max_val = max_val;
        out_img->data = malloc(width * height);
    }

    MPI_Gatherv(local_out, local_rows * width, MPI_UNSIGNED_CHAR,
                rank == 0 ? out_img->data : NULL, sendcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        int num_threads = 1;
#pragma omp parallel
        {
#pragma omp single
            num_threads = omp_get_num_threads();
        }
        printf("Tempo di convoluzione IBRIDA (%d proc MPI x %d thread OMP): %.4f secondi\n",
               size, num_threads, end_time - start_time);

        write_pgm(output_file, out_img);
        free(out_img->data);
        free(out_img);
        free(img->data);
        free(img);
        free(sendcounts);
        free(displs);
    }

    free(kernel);
    free(local_data);
    free(local_out);
    free(padded_data);
    MPI_Finalize();
    return 0;
}