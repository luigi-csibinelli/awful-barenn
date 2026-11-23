#pragma once
// Partes comuns às versões CPU (barenn.cpp) e GPU (barenn.cu): camada densa,
// leitura dos dados, perda e avaliação.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // Pesos contíguos e TRANSPOSTOS: weights[j * out_features + i] liga a
    // entrada j ao neurônio de saída i. Assim, para uma entrada j fixa, os pesos
    // de todos os neurônios ficam lado a lado na memória.
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes. A entrada não
    // é copiada: guardamos apenas o endereço, que segue válido até o backward.
    const float* last_input = nullptr;
    vector<float> last_z;
    // Buffers alocados uma única vez e reutilizados a cada amostra.
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(size_t(output_size) * input_size), bias(output_size, 0),
          last_z(output_size), out(output_size), delta(output_size),
          grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // Os números são sorteados na mesma ordem da versão original
        // (neurônio i por fora, entrada j por dentro).
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[size_t(j) * out_features + i] = distribution(gen);
            }
        }
    }

    const vector<float>& Forward(const vector<float>& input) {
        last_input = input.data();
        // Laço invertido (entrada j por fora, neurônio i por dentro): cada
        // last_z[i] recebe as parcelas na mesma ordem j = 0, 1, ... do código
        // original, mas os neurônios agora são independentes entre si no laço
        // interno, que percorre a memória em sequência e pode ser vetorizado.
        for (int i = 0; i < out_features; ++i) {
            last_z[i] = bias[i];
        }
        for (int j = 0; j < in_features; ++j) {
            const float x = input[j];
            const float* w = &weights[size_t(j) * out_features];
            for (int i = 0; i < out_features; ++i) {
                last_z[i] += x * w[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            out[i] = final ? last_z[i] : max(0.0f, last_z[i]);
        }
        return out;
    }

    // calcular_grad_in = false pula o gradiente da entrada na primeira camada,
    // onde ele não é usado por ninguém.
    const vector<float>& Backward(const vector<float>& grad_out,
                                  float learning_rate,
                                  bool calcular_grad_in = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        for (int j = 0; j < in_features; ++j) {
            float* w = &weights[size_t(j) * out_features];
            // O gradiente da entrada usa os pesos anteriores à atualização,
            // somados na ordem i = 0, 1, ... como no código original.
            if (calcular_grad_in) {
                float sum = 0;
                for (int i = 0; i < out_features; ++i) {
                    sum += delta[i] * w[i];
                }
                grad_in[j] = sum;
            }
            // SGD por amostra: laço contíguo e sem dependência entre iterações.
            const float x = last_input[j];
            for (int i = 0; i < out_features; ++i) {
                w[i] -= learning_rate * (delta[i] * x);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= learning_rate * delta[i];
        }
        return grad_in;
    }

    // ---------------------------------------------------------------------
    // Modo mini-batch (OpenMP). As funções abaixo NÃO alteram a camada: todas
    // as amostras do lote usam os mesmos pesos, então podem ser processadas
    // ao mesmo tempo por threads diferentes, cada uma com seus próprios
    // buffers (padrão "private + merge" da Aula 8).
    // ---------------------------------------------------------------------
    size_t NumPesos() const { return weights.size(); }
    // Acesso direto aos parâmetros (usado na atualização do lote e na GPU).
    vector<float>& Pesos() { return weights; }
    vector<float>& Vies() { return bias; }

    void ForwardAmostra(const float* input, float* z,
                        float* a) const {
        const float* wt = weights.data();
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        for (int j = 0; j < in_features; ++j) {
            const float x = input[j];
            const float* w = wt + size_t(j) * out_features;
            for (int i = 0; i < out_features; ++i) {
                z[i] += x * w[i];
            }
        }
        for (int i = 0; i < out_features; ++i) {
            a[i] = final ? z[i] : max(0.0f, z[i]);
        }
    }

    // Calcula delta e o gradiente da entrada (se grad_in != nullptr) e SOMA o
    // gradiente dos pesos/viés desta amostra no acumulador da thread (gW, gb).
    void BackwardAmostra(const float* input,
                         const float* z,
                         const float* grad_out,
                         float* delta, float* grad_in,
                         float* gW, float* gb) const {
        const float* wt = weights.data();
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        }
        for (int j = 0; j < in_features; ++j) {
            const float* w = wt + size_t(j) * out_features;
            if (grad_in) {
                float sum = 0;
                for (int i = 0; i < out_features; ++i) {
                    sum += delta[i] * w[i];
                }
                grad_in[j] = sum;
            }
            const float x = input[j];
            float* g = gW + size_t(j) * out_features;
            for (int i = 0; i < out_features; ++i) {
                g[i] += delta[i] * x;
            }
        }
        for (int i = 0; i < out_features; ++i) {
            gb[i] += delta[i];
        }
    }

    // Soma de todos os parâmetros, na mesma ordem da versão original.
    // Usada apenas para validar a corretude.
    double Checksum() const {
        double total = 0;
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                total += weights[size_t(j) * out_features + i];
            }
        }
        for (float b : bias) {
            total += b;
        }
        return total;
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

inline vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        amostra.imagem.reserve(entradas);
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
inline float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

inline void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn
