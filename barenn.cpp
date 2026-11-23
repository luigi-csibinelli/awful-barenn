#include "barenn.hpp"

#include <omp.h>

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
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Tamanho do lote: variável de ambiente BARENN_BATCH (padrão 1).
        // Com B = 1 o algoritmo é exatamente o SGD por amostra original.
        int B = 1;
        if (const char* valor = getenv("BARENN_BATCH")) {
            B = max(1, atoi(valor));
        }
        const int T = omp_get_max_threads();
        cerr << "mini-batch B = " << B << " | threads = " << T << "\n";

        // Acumulador de gradientes de cada thread: [W1 b1 W2 b2 W3 b3].
        const int n1 = 32, n2 = 16, n3 = Barenn::classes;
        const size_t oW1 = 0, ob1 = oW1 + l1.NumPesos();
        const size_t oW2 = ob1 + n1, ob2 = oW2 + l2.NumPesos();
        const size_t oW3 = ob2 + n2, ob3 = oW3 + l3.NumPesos();
        const size_t P = ob3 + n3;
        vector<float> acumuladores(size_t(T) * P);
        // destino[q] aponta para o parâmetro que corresponde à posição q do
        // acumulador. Assim a atualização das três camadas vira um único laço.
        vector<float*> destino(P);
        for (size_t k = 0; k < l1.NumPesos(); ++k) destino[oW1 + k] = &l1.Pesos()[k];
        for (int k = 0; k < n1; ++k) destino[ob1 + k] = &l1.Vies()[k];
        for (size_t k = 0; k < l2.NumPesos(); ++k) destino[oW2 + k] = &l2.Pesos()[k];
        for (int k = 0; k < n2; ++k) destino[ob2 + k] = &l2.Vies()[k];
        for (size_t k = 0; k < l3.NumPesos(); ++k) destino[oW3 + k] = &l3.Pesos()[k];
        for (int k = 0; k < n3; ++k) destino[ob3 + k] = &l3.Vies()[k];
        // Perda acumulada por thread.
        vector<double> perda_thread(T, 0.0);
        const int N = static_cast<int>(treino.size());

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        // Uma única região paralela para todo o treino: as threads são criadas
        // uma vez (fork/join) e reaproveitadas em todos os lotes.
        #pragma omp parallel
        {
            const int t = omp_get_thread_num();
            float* meu = &acumuladores[size_t(t) * P];
            // Buffers privados da thread: ela processa uma amostra por vez.
            vector<float> z1(n1), a1(n1), d1(n1), g1(n1);
            vector<float> z2(n2), a2(n2), d2(n2), g2(n2);
            vector<float> z3(n3), a3(n3), d3(n3), grad3(n3);

            for (int epoca = 0; epoca < epocas; ++epoca) {
                // Uma época percorre todo o conjunto na ordem do arquivo.
                for (int base = 0; base < N; base += B) {
                    const int nb = min(B, N - base);
                    fill(meu, meu + P, 0.0f);

                    // 1) Cada thread processa amostras diferentes do lote com
                    //    os MESMOS pesos e acumula os gradientes no seu buffer.
                    #pragma omp for schedule(static)
                    for (int b = 0; b < nb; ++b) {
                        const auto& amostra = treino[base + b];
                        const float* x = amostra.imagem.data();
                        l1.ForwardAmostra(x, z1.data(), a1.data());
                        l2.ForwardAmostra(a1.data(), z2.data(), a2.data());
                        l3.ForwardAmostra(a2.data(), z3.data(), a3.data());
                        perda_thread[t] +=
                            Barenn::perda(a3, amostra.rotulo, &grad3);
                        l3.BackwardAmostra(a2.data(), z3.data(), grad3.data(),
                                           d3.data(), g2.data(),
                                           meu + oW3, meu + ob3);
                        l2.BackwardAmostra(a1.data(), z2.data(), g2.data(),
                                           d2.data(), g1.data(),
                                           meu + oW2, meu + ob2);
                        l1.BackwardAmostra(x, z1.data(), g1.data(), d1.data(),
                                           nullptr, meu + oW1, meu + ob1);
                    }
                    // (barreira implícita do omp for: todos os gradientes prontos)

                    // 2) Merge: soma os acumuladores das T threads (na ordem
                    //    k = 0, 1, ...) e atualiza os parâmetros. O omp for divide
                    //    os parâmetros entre as threads: cada posição é escrita
                    //    por uma única thread (sem condição de corrida). Um laço
                    //    só para as três camadas => uma barreira só.
                    #pragma omp for schedule(static)
                    for (size_t q = 0; q < P; ++q) {
                        float g = acumuladores[q];
                        for (int k = 1; k < T; ++k) {
                            g += acumuladores[size_t(k) * P + q];
                        }
                        *destino[q] -= taxa * g;
                    }
                }
                // Fim da época: soma as perdas das threads em ordem fixa.
                #pragma omp single
                {
                    double perda_total = 0;
                    for (int k = 0; k < T; ++k) {
                        perda_total += perda_thread[k];
                        perda_thread[k] = 0;
                    }
                    perdas[epoca] = perda_total / treino.size();
                }
            }
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
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
