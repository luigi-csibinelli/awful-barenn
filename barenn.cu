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

// Forward de uma camada: uma thread por neurônio de saída i. A soma segue a
// ordem j = 0, 1, ... como na CPU. Os pesos estão transpostos
// (W[j * out + i]), então threads vizinhas leem endereços vizinhos
// (acesso coalescido à memória global).
__global__ void forward_camada(const float* x, const float* W, const float* b,
                               float* z, float* a, int in, int out,
                               bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out) {
        float soma = b[i];
        for (int j = 0; j < in; ++j) {
            soma += x[j] * W[j * out + i];
        }
        z[i] = soma;
        a[i] = final ? soma : (0.0f < soma ? soma : 0.0f);
    }
}

// Backward de uma camada: a thread j cuida da linha j da matriz transposta.
// Ela soma o gradiente da entrada com os pesos ANTIGOS (ordem i = 0, 1, ...)
// e depois atualiza a sua linha. As threads j < out também atualizam o viés.
__global__ void backward_camada(const float* x, const float* z,
                                const float* grad_out, float* W, float* b,
                                float* grad_in, int in, int out, bool final,
                                float taxa) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < in) {
        float* w = W + j * out;
        const float xj = x[j];
        float soma = 0;
        for (int i = 0; i < out; ++i) {
            const float delta =
                final ? grad_out[i]
                      : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
            if (grad_in) {
                soma += delta * w[i];
            }
            w[i] -= taxa * (delta * xj);
        }
        if (grad_in) {
            grad_in[j] = soma;
        }
    }
    if (j < out) {
        const float delta =
            final ? grad_out[j] : grad_out[j] * (z[j] > 0.0f ? 1.0f : 0.0f);
        b[j] -= taxa * delta;
    }
}

// Buffers de uma camada na GPU, alocados uma única vez.
struct CamadaGPU {
    int in = 0, out = 0;
    bool final = false;
    float *W = nullptr, *b = nullptr, *z = nullptr, *a = nullptr;
    float* grad_in = nullptr;

    CamadaGPU(Barenn::DenseLayer& camada, int entradas, int saidas,
              bool ultima)
        : in(entradas), out(saidas), final(ultima) {
        cudaMalloc(&W, sizeof(float) * in * out);
        cudaMalloc(&b, sizeof(float) * out);
        cudaMalloc(&z, sizeof(float) * out);
        cudaMalloc(&a, sizeof(float) * out);
        cudaMalloc(&grad_in, sizeof(float) * in);
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
        cudaFree(z);
        cudaFree(a);
        cudaFree(grad_in);
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

        CamadaGPU g1(l1, Barenn::entradas, 32, false);
        CamadaGPU g2(l2, 32, 16, false);
        CamadaGPU g3(l3, 16, Barenn::classes, true);
        float* d_gradiente = nullptr;
        cudaMalloc(&d_gradiente, sizeof(float) * Barenn::classes);
        vector<float> escores(Barenn::classes), gradiente(Barenn::classes);

        verificar_cuda("preparacao");

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (int n = 0; n < N; ++n) {
                const float* x = d_imagens + size_t(n) * Barenn::entradas;
                // Forward: um kernel por camada.
                forward_camada<<<1, bloco>>>(x, g1.W, g1.b, g1.z, g1.a,
                                             g1.in, g1.out, false);
                forward_camada<<<1, bloco>>>(g1.a, g2.W, g2.b, g2.z, g2.a,
                                             g2.in, g2.out, false);
                forward_camada<<<1, bloco>>>(g2.a, g3.W, g3.b, g3.z, g3.a,
                                             g3.in, g3.out, true);
                // A perda ainda é calculada na CPU: os escores vão para o
                // host e o gradiente volta para a GPU a cada amostra.
                cudaMemcpy(escores.data(), g3.a,
                           sizeof(float) * Barenn::classes,
                           cudaMemcpyDeviceToHost);
                perda_total +=
                    Barenn::perda(escores, treino[n].rotulo, &gradiente);
                cudaMemcpy(d_gradiente, gradiente.data(),
                           sizeof(float) * Barenn::classes,
                           cudaMemcpyHostToDevice);
                // Backward: um kernel por camada (a primeira não precisa do
                // gradiente da entrada).
                backward_camada<<<1, bloco>>>(g2.a, g3.z, d_gradiente, g3.W,
                                              g3.b, g3.grad_in, g3.in, g3.out,
                                              true, taxa);
                backward_camada<<<1, bloco>>>(g1.a, g2.z, g3.grad_in, g2.W,
                                              g2.b, g2.grad_in, g2.in, g2.out,
                                              false, taxa);
                backward_camada<<<1, bloco>>>(x, g1.z, g2.grad_in, g1.W, g1.b,
                                              nullptr, g1.in, g1.out, false,
                                              taxa);
            }
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
        cudaFree(d_gradiente);

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
