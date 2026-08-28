#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
    int width;
    int height;
    int max_val;
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
        fprintf(stderr, "Errore: impossibile aprire %s\n", filename);
        exit(1);
    }

    PGMImage *img = (PGMImage *)malloc(sizeof(PGMImage));
    char format[3];

    if (fscanf(file, "%2s", format) != 1)
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

    if (fscanf(file, "%d %d", &img->width, &img->height) != 2)
    {
        fprintf(stderr, "Errore: impossibile leggere le dimensioni dell'immagine.\n");
        exit(1);
    }

    if (fscanf(file, "%d", &img->max_val) != 1)
    {
        fprintf(stderr, "Errore: impossibile leggere il valore massimo.\n");
        exit(1);
    }

    fgetc(file);

    int size = img->width * img->height;
    img->data = (unsigned char *)malloc(size * sizeof(unsigned char));

    if (fread(img->data, sizeof(unsigned char), size, file) != (size_t)size)
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

    if (fwrite(img->data, sizeof(unsigned char), img->width * img->height, file) != (size_t)(img->width * img->height))
    {
        fprintf(stderr, "Errore: scrittura dei pixel fallita.\n");
        exit(1);
    }

    fclose(file);
}

void free_pgm(PGMImage *img)
{
    if (img)
    {
        free(img->data);
        free(img);
    }
}

float *generate_gaussian_kernel(int k_size)
{
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));
    float sum = 0.0f;
    int r = k_size / 2;
    float sigma = 1.0f;
    float s = 2.0f * sigma * sigma;

    for (int x = -r; x <= r; x++)
    {
        for (int y = -r; y <= r; y++)
        {
            float r_squared = x * x + y * y;
            float val = (exp(-(r_squared) / s)) / (M_PI * s);
            kernel[(x + r) * k_size + (y + r)] = val;
            sum += val;
        }
    }

    for (int i = 0; i < k_size * k_size; i++)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

PGMImage *convolution_omp(PGMImage *input, float *kernel, int k_size)
{
    PGMImage *output = (PGMImage *)malloc(sizeof(PGMImage));
    output->width = input->width;
    output->height = input->height;
    output->max_val = input->max_val;
    output->data = (unsigned char *)malloc(output->width * output->height * sizeof(unsigned char));

    int r = k_size / 2;

// Parallelizzazione del ciclo esterno
#pragma omp parallel for
    for (int y = 0; y < input->height; y++)
    {
        for (int x = 0; x < input->width; x++)
        {
            float sum = 0.0f;

            for (int ky = -r; ky <= r; ky++)
            {
                for (int kx = -r; kx <= r; kx++)
                {
                    int px = clamp(x + kx, 0, input->width);
                    int py = clamp(y + ky, 0, input->height);

                    float weight = kernel[(ky + r) * k_size + (kx + r)];
                    unsigned char pixel_val = input->data[py * input->width + px];

                    sum += pixel_val * weight;
                }
            }

            int final_pixel = (int)sum;
            if (final_pixel < 0)
                final_pixel = 0;
            if (final_pixel > 255)
                final_pixel = 255;

            output->data[y * input->width + x] = (unsigned char)final_pixel;
        }
    }

    return output;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Uso: %s <input.pgm> <output.pgm> <kernel_size>\n", argv[0]);
        printf("Nota: kernel_size deve essere dispari e <= 9 (es. 3, 5, 7, 9)\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int k_size = atoi(argv[3]);

    if (k_size % 2 == 0 || k_size > 9 || k_size < 3)
    {
        fprintf(stderr, "Errore: La dimensione del kernel deve essere un numero dispari compreso tra 3 e 9.\n");
        return 1;
    }

    int num_threads = 0;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    printf("Inizio elaborazione OpenMP (%d thread)...\n", num_threads);
    printf("- Immagine: %s\n", input_file);
    printf("- Kernel: %dx%d\n", k_size, k_size);

    PGMImage *img = read_pgm(input_file);
    float *kernel = generate_gaussian_kernel(k_size);

    // ---------------- TIMER START --------------
    double start_time = omp_get_wtime();

    // Esecuzione
    PGMImage *out_img = convolution_omp(img, kernel, k_size);

    // ---------------- TIMER STOP ----------------
    double end_time = omp_get_wtime();

    printf("Tempo di convoluzione OpenMP: %.4f secondi\n", end_time - start_time);

    write_pgm(output_file, out_img);

    free(kernel);
    free_pgm(img);
    free_pgm(out_img);

    printf("Filtro applicato con successo. Salvato in %s\n", output_file);

    return 0;
}