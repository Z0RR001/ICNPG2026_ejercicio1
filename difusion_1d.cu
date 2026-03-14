%%writefile difusion_1d.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpu_timer.h"

// Kernel que implementa el stencil de diferencias finitas 1D
__global__ void diffusionKernel(float *u_old, float *u_new, float r, int L) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < L-1)
    {
        u_new[i] = u_old[i] + r*(u_old[i+1] - 2*u_old[i] + u_old[i-1]);
    }
}

int main() {
    // 1. Configuración de parámetros
    const int L = 512;            // Puntos en la barra
    const int steps = 20000;      // Pasos temporales
    const int output_freq = 100;  // Frecuencia de muestreo del centro
    const float r = 0.4f;         // Coeficiente de estabilidad (r <= 0.5)

    size_t size = L * sizeof(float);

    // 2. Inicialización del perfil (Pulso Cuadrado)
    std::vector<float> h_u(L, 0.0f);
   // for (int i = L / 4; i < 3 * L / 4; ++i) {
   //     h_u[i] = 1.0f;
   // }

    for(int i=0;i<L;i++)
    {

      if(i > L/4 && i < 3*L/4)
          h_u[i] = 1.0;
      else
          h_u[i] = 0.0;
    }

    // 3. Reserva de memoria en GPU y Double Buffering
    float *d_u_old, *d_u_new;
    cudaMalloc(&d_u_old, size);
    cudaMalloc(&d_u_new, size);

    // Copiar condición inicial y limpiar el segundo buffer
    cudaMemcpy(d_u_old, h_u.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_u_new, 0, size);

    // 4. Parámetros de ejecución de CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (L + threadsPerBlock - 1) / threadsPerBlock;


    std::ofstream archivo("log.txt");
    std::printf("# Paso \t Temperatura_Centro\n");
     archivo << "Tiempo_(ms) \t Temperatura_Centro" << "\n";


    // Agregar timers de GPU para medir el tiempo de ejecución de todo el loop
    gpu_timer Reloj;
	  Reloj.tic();

    // 5. Bucle de evolución temporal
    for (int s = 0; s <= steps; ++s) {

        // --- OPTIMIZACIÓN DE TRANSFERENCIA ---
        // Extraemos solo el punto central del buffer actual antes del swap
        if (s % output_freq == 0 && s != 0) {                                        //agregue "&& s != 0" porque sino me imprimía  el primer valor de  temperatura = 0
            // Se transfiere un único float (4 bytes) evitando copiar todo el array
            float center;
            cudaMemcpy(&center, &d_u_new[L/2], sizeof(float), cudaMemcpyDeviceToHost);

            // Se imprime ese único float en el host
            printf("%d \t %f\n", s/output_freq, center);
            archivo << Reloj.tac() << "\t \t \t \t \t" << center << "\n";
        }

        // Lanzamiento del kernel
        diffusionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_u_old, d_u_new, r, L);

        // --- GESTIÓN DE MEMORIA (Swap de punteros) ---
        // Intercambiamos los buffers: el "nuevo" pasa a ser el "viejo" para el siguiente paso
        float *temp = d_u_old;
        d_u_old = d_u_new;
        d_u_new = temp;
    }

    // Imprimir el tiempo de ejecución de todo el loop
    printf("\nTiempo de ejecución del loop= %lf ms\n", Reloj.tac());

    // 6. Liberación de memoria
    cudaFree(d_u_old);
    cudaFree(d_u_new);

    // 7. Cierre del archivo de salida
    archivo.close();

    return 0;
}
