#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Struttura per l'immagine PGM
typedef struct
{
    int width;
    int height;
    int max_val;
    unsigned char *data;
} PGMImage;

// Funzione di utilità per il CLAMPING
// Mantiene le coordinate (x, y) all'interno dei bordi dell'immagine
int clamp(int val, int min, int max)
{
    if (val < min)
        return min;
    if (val >= max)
        return max - 1;
    return val;
}

// Funzione per leggere un'immagine PGM (formato binario P5)
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
    fscanf(file, "%2s", format);
    if (strcmp(format, "P5") != 0)
    {
        fprintf(stderr, "Errore: il file non è un PGM binario (P5)\n");
        exit(1);
    }

    // Salta i commenti
    int c = getc(file);
    while (c == '#' || c == '\n' || c == ' ' || c == '\r')
    {
        if (c == '#')
        {
            while (getc(file) != '\n')
                ; // Salta fino a fine riga
        }
        else
        {
            c = getc(file);
        }
    }
    ungetc(c, file);

    // Leggi dimensioni e valore massimo
    fscanf(file, "%d %d", &img->width, &img->height);
    fscanf(file, "%d", &img->max_val);
    fgetc(file); // Consuma il carattere newline dopo il max_val

    // Alloca memoria e leggi i pixel
    int size = img->width * img->height;
    img->data = (unsigned char *)malloc(size * sizeof(unsigned char));
    fread(img->data, sizeof(unsigned char), size, file);

    fclose(file);
    return img;
}

// Funzione per salvare un'immagine PGM
void write_pgm(const char *filename, PGMImage *img)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Errore: impossibile creare %s\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img->width, img->height, img->max_val);
    fwrite(img->data, sizeof(unsigned char), img->width * img->height, file);
    fclose(file);
}

// Libera la memoria
void free_pgm(PGMImage *img)
{
    if (img)
    {
        free(img->data);
        free(img);
    }
}

// Genera un kernel Gaussiano 1D/2D
// La dimensione 'k_size' deve essere dispari (es. 3, 5, 7, 9)
float *generate_gaussian_kernel(int k_size)
{
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));
    float sum = 0.0f;
    int r = k_size / 2;
    float sigma = 1.0f; // Modifica questo valore per variare l'intensità della sfocatura (blur)
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

    // Normalizza il kernel in modo che la somma di tutti gli elementi sia 1.0
    // Questo previene che l'immagine diventi più chiara o più scura
    for (int i = 0; i < k_size * k_size; i++)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

// Convoluzione Sequenziale (Baseline)
PGMImage *convolution_seq(PGMImage *input, float *kernel, int k_size)
{
    PGMImage *output = (PGMImage *)malloc(sizeof(PGMImage));
    output->width = input->width;
    output->height = input->height;
    output->max_val = input->max_val;
    output->data = (unsigned char *)malloc(output->width * output->height * sizeof(unsigned char));

    int r = k_size / 2;

#pragma omp parallel for
    for (int y = 0; y < input->height; y++)
    {
        for (int x = 0; x < input->width; x++)
        {
            float sum = 0.0f;

            // Applica il kernel
            for (int ky = -r; ky <= r; ky++)
            {
                for (int kx = -r; kx <= r; kx++)
                {
                    // Usa la funzione clamp per gestire i bordi (evita segmentation fault)
                    int px = clamp(x + kx, 0, input->width);
                    int py = clamp(y + ky, 0, input->height);

                    float weight = kernel[(ky + r) * k_size + (kx + r)];
                    unsigned char pixel_val = input->data[py * input->width + px];

                    sum += pixel_val * weight;
                }
            }

            // Assicuriamoci che il valore finale sia compreso tra 0 e 255
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

// Main aggiornato per leggere gli argomenti
int main(int argc, char *argv[])
{
    // Controllo argomenti
    if (argc != 4)
    {
        printf("Uso: %s <input.pgm> <output.pgm> <kernel_size>\n", argv[0]);
        printf("Nota: kernel_size deve essere dispari e <= 9 (es. 3, 5, 7, 9)\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int k_size = atoi(argv[3]);

    // Validazione dimensione kernel
    if (k_size % 2 == 0 || k_size > 9 || k_size < 3)
    {
        fprintf(stderr, "Errore: La dimensione del kernel deve essere un numero dispari compreso tra 3 e 9.\n");
        return 1;
    }

    printf("Inizio elaborazione sequenziale...\n");
    printf("- Immagine: %s\n", input_file);
    printf("- Kernel: %dx%d\n", k_size, k_size);

    PGMImage *img = read_pgm(input_file);
    float *kernel = generate_gaussian_kernel(k_size);

    // ---------------- TIMER START ----------------
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Esecuzione della convoluzione
    PGMImage *out_img = convolution_seq(img, kernel, k_size);

    // ---------------- TIMER STOP ----------------
    gettimeofday(&end, NULL);

    // Calcola il tempo in secondi
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Tempo di convoluzione: %.4f secondi\n", time_spent);

    write_pgm(output_file, out_img);

    free(kernel);
    free_pgm(img);
    free_pgm(out_img);

    printf("Filtro applicato con successo. Salvato in %s\n", output_file);

    return 0;
}