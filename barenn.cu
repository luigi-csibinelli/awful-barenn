// Versão GPU (CUDA) do treino. A leitura dos dados, a perda e a avaliação
// final reaproveitam o código da CPU (barenn.hpp).
#include "barenn.hpp"

#include <cuda_runtime.h>

// Verifica se alguma chamada CUDA anterior falhou (cudaGetLastError, como
// nos exemplos da aula) e interrompe o programa com a mensagem do erro.
void verificar_cuda(const string& etapa) {
    cudaError_t erro = cudaGetLastError();
    if (erro != cudaSuccess) {
        throw runtime_error("Erro CUDA (" + etapa + "): " +
                            cudaGetErrorString(erro));
    }
}

// Dimensões da rede 64 -> 32 -> 16 -> 10.
constexpr int E = Barenn::entradas, H1 = 32, H2 = 16, C = Barenn::classes;
// Forward de uma camada: a thread i calcula o neurônio i somando na ordem
// j = 0, 1, ... (igual à CPU). "passo" é a distância entre duas linhas da
// matriz transposta de pesos.
__device__ void forward(const float* x, const float* W, int passo,
                        const float* b, float* z, float* a, int in, int out,
                        bool final) {
    for (int i = threadIdx.x; i < out; i += blockDim.x) {
        float soma = b[i];
        for (int j = 0; j < in; ++j) {
            soma += x[j] * W[j * passo + i];
        }
        z[i] = soma;
        if (a) {
            a[i] = final ? soma : (0.0f < soma ? soma : 0.0f);
        }
    }
}

// Backward de uma camada: a thread j soma o gradiente da entrada com os
// pesos ANTIGOS (ordem i = 0, 1, ...) e atualiza a linha j. As threads
// j < out atualizam o viés.
__device__ void backward(const float* x, const float* delta, float* W,
                         int passo, float* b, float* grad_in, int in, int out,
                         float taxa) {
    for (int j = threadIdx.x; j < in; j += blockDim.x) {
        float* w = W + j * passo;
        const float xj = x[j];
        float soma = 0;
        for (int i = 0; i < out; ++i) {
            if (grad_in) {
                soma += delta[i] * w[i];
            }
            w[i] -= taxa * (delta[i] * xj);
        }
        if (grad_in) {
            grad_in[j] = soma;
        }
    }
    for (int i = threadIdx.x; i < out; i += blockDim.x) {
        b[i] -= taxa * delta[i];
    }
}

// Derivada da ReLU aplicada ao gradiente que chega na camada oculta.
__device__ void delta_relu(const float* grad, const float* z, float* delta,
                           int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        delta[i] = grad[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// Treina UMA época inteira em um único lançamento, com um único bloco.
// O SGD continua amostra por amostra (cada amostra depende dos pesos
// atualizados pela anterior); o paralelismo está DENTRO de cada amostra,
// entre os neurônios. Nesta versão tudo fica na memória global: pesos,
// vieses e um buffer de trabalho com as ativações e os gradientes.
__global__ void treinar_epoca(const float* imagens, const int* rotulos, int N,
                              float* W1, float* b1, float* W2, float* b2,
                              float* W3, float* b3, float taxa,
                              double* perda_epoca, float* trabalho) {
    float* z1 = trabalho;
    float* a1 = z1 + H1;
    float* z2 = a1 + H1;
    float* a2 = z2 + H2;
    float* z3 = a2 + H2;
    float* g3 = z3 + C;
    float* g2 = g3 + C;
    float* g1 = g2 + H2;
    float* d2 = g1 + H1;
    float* d1 = d2 + H2;
    double perda_total = 0;  // usada só pela thread 0

    for (int n = 0; n < N; ++n) {
        const float* x = imagens + n * E;
        forward(x, W1, H1, b1, z1, a1, E, H1, false);
        __syncthreads();
        forward(a1, W2, H2, b2, z2, a2, H1, H2, false);
        __syncthreads();
        forward(a2, W3, C, b3, z3, nullptr, H2, C, true);
        __syncthreads();

        // Perda (metade da soma dos erros quadráticos), igual à CPU. São só
        // 10 termos: uma thread calcula, na mesma ordem da versão original.
        if (threadIdx.x == 0) {
            const int rotulo = rotulos[n];
            float total = 0;
            for (int i = 0; i < C; ++i) {
                const float erro = z3[i] - (i == rotulo ? 1.0f : 0.0f);
                total += 0.5f * erro * erro;
                g3[i] = erro;
            }
            perda_total += total;
        }
        __syncthreads();

        // Camada 3 (linear): delta é o próprio gradiente da perda.
        backward(a2, g3, W3, C, b3, g2, H2, C, taxa);
        __syncthreads();
        delta_relu(g2, z2, d2, H2);
        __syncthreads();
        backward(a1, d2, W2, H2, b2, g1, H1, H2, taxa);
        __syncthreads();
        delta_relu(g1, z1, d1, H1);
        __syncthreads();
        backward(x, d1, W1, H1, b1, nullptr, E, H1, taxa);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *perda_epoca = perda_total;
    }
}

// Buffers de uma camada na GPU, alocados uma única vez.
struct CamadaGPU {
    int in = 0, out = 0;
    bool final = false;
    float *W = nullptr, *b = nullptr;

    CamadaGPU(Barenn::DenseLayer& camada, int entradas, int saidas,
              bool ultima)
        : in(entradas), out(saidas), final(ultima) {
        cudaMalloc(&W, sizeof(float) * in * out);
        cudaMalloc(&b, sizeof(float) * out);
        cudaMemcpy(W, camada.Pesos().data(), sizeof(float) * in * out,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(b, camada.Vies().data(), sizeof(float) * out,
                   cudaMemcpyHostToDevice);
    }
    // Copia os parâmetros treinados de volta para a camada da CPU.
    void CopiarPara(Barenn::DenseLayer& camada) const {
        cudaMemcpy(camada.Pesos().data(), W, sizeof(float) * in * out,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(camada.Vies().data(), b, sizeof(float) * out,
                   cudaMemcpyDeviceToHost);
    }
    ~CamadaGPU() {
        cudaFree(W);
        cudaFree(b);
    }
};

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        constexpr int bloco = 64;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // O conjunto de treino vai para a GPU uma única vez.
        const int N = static_cast<int>(treino.size());
        vector<float> imagens(size_t(N) * Barenn::entradas);
        for (int n = 0; n < N; ++n) {
            copy(treino[n].imagem.begin(), treino[n].imagem.end(),
                 imagens.begin() + size_t(n) * Barenn::entradas);
        }
        float* d_imagens = nullptr;
        cudaMalloc(&d_imagens, sizeof(float) * imagens.size());
        cudaMemcpy(d_imagens, imagens.data(),
                   sizeof(float) * imagens.size(),
                   cudaMemcpyHostToDevice);

        vector<int> rotulos(N);
        for (int n = 0; n < N; ++n) {
            rotulos[n] = treino[n].rotulo;
        }
        int* d_rotulos = nullptr;
        cudaMalloc(&d_rotulos, sizeof(int) * N);
        cudaMemcpy(d_rotulos, rotulos.data(), sizeof(int) * N,
                   cudaMemcpyHostToDevice);

        CamadaGPU g1(l1, Barenn::entradas, 32, false);
        CamadaGPU g2(l2, 32, 16, false);
        CamadaGPU g3(l3, 16, Barenn::classes, true);
        double* d_perda = nullptr;
        cudaMalloc(&d_perda, sizeof(double));
        // Ativações e gradientes intermediários de uma amostra (212 floats).
        float* d_trabalho = nullptr;
        cudaMalloc(&d_trabalho, sizeof(float) * 256);

        verificar_cuda("preparacao");

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            // Uma época inteira em um único kernel.
            treinar_epoca<<<1, bloco>>>(d_imagens, d_rotulos, N, g1.W, g1.b,
                                        g2.W, g2.b, g3.W, g3.b, taxa, d_perda,
                                        d_trabalho);
            double perda_total = 0;
            cudaMemcpy(&perda_total, d_perda, sizeof(double),
                       cudaMemcpyDeviceToHost);
            perdas[epoca] = perda_total / treino.size();
        }
        cudaDeviceSynchronize();
        verificar_cuda("treino");
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Os pesos treinados voltam para a CPU para a avaliação final.
        g1.CopiarPara(l1);
        g2.CopiarPara(l2);
        g3.CopiarPara(l3);
        verificar_cuda("copia dos pesos");
        cudaFree(d_imagens);
        cudaFree(d_rotulos);
        cudaFree(d_perda);
        cudaFree(d_trabalho);

        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        cout << "Checksum dos pesos " << setprecision(9)
             << l1.Checksum() + l2.Checksum() + l3.Checksum() << "\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
