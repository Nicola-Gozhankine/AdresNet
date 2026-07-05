#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include "adresnet.cpp"

// ---------- Параметры ----------
const int NUM_GENOTYPES = 100000;       // сколько генотипов проверить
const int MIN_NEURONS = 20;
const int MAX_NEURONS = 80;
const int MAX_STEPS = 25;               // шагов на один входной бит
const int INPUT_NEURON = 0;
const int NONCE_BITS = 16;              // для инициализации (не используется, но оставим)
const int NONCE = 12345;

// Файлы с обучающей и валидационной последовательностью
const std::string TRAIN_FILE = "train.bin";
const std::string VALID_FILE = "valid.bin";

// ---------- Генерация случайного генотипа (все в слое 0) ----------
std::vector<std::vector<int>> random_genotype(int num_neurons) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> type_dist(0, 1);
    std::uniform_int_distribution<int> mode_dist(0, 1);
    std::uniform_int_distribution<int> action_dist(0, 4);
    std::uniform_int_distribution<int> local_dist(0, num_neurons - 1);

    std::vector<std::vector<int>> genotype;
    genotype.push_back({num_neurons});

    for (int i = 0; i < num_neurons; ++i) {
        std::vector<int> block(19, 0);
        block[0] = 0; // слой 0
        block[1] = type_dist(gen);
        block[2] = mode_dist(gen);
        // fixed0 и fixed1 – цели в слое 0
        block[3] = 0;
        block[4] = local_dist(gen);
        block[5] = 0;
        block[6] = local_dist(gen);
        // адресные регистры (delta = 0, целевой слой 0)
        for (int j = 0; j < 4; ++j) {
            block[7 + j*2] = 0;
            block[8 + j*2] = local_dist(gen);
        }
        // действия
        for (int j = 0; j < 4; ++j) {
            block[15 + j] = action_dist(gen);
        }
        genotype.push_back(block);
    }
    return genotype;
}

// ---------- Загрузка бинарного файла ----------
std::vector<uint8_t> load_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
}

// ---------- Оценка точности предсказания следующего байта ----------
double evaluate_prediction(Network& net, const std::vector<uint8_t>& data,
                           int max_steps, int input_neuron) {
    net.reset(); // сброс состояний в 0
    int net_size = net.neurons.size();
    if (net_size < 8) return 0.0; // недостаточно выходов

    // Выходные нейроны – последние 8
    std::vector<int> output_neurons(8);
    for (int i = 0; i < 8; ++i) {
        output_neurons[i] = net_size - 8 + i;
    }

    int correct = 0;
    int total = data.size() - 1; // предсказываем следующий байт для каждого, кроме последнего
    for (size_t i = 0; i < data.size() - 1; ++i) {
        uint8_t current_byte = data[i];
        // Подаём 8 бит текущего байта (MSB first)
        for (int bit = 7; bit >= 0; --bit) {
            int bit_val = (current_byte >> bit) & 1;
            net.external_input(input_neuron, bit_val);
            for (int step = 0; step < max_steps; ++step) {
                net.step();
            }
        }
        // После обработки байта считываем выходные нейроны
        uint8_t predicted_byte = 0;
        for (int b = 0; b < 8; ++b) {
            predicted_byte = (predicted_byte << 1) | net.neurons[output_neurons[b]]->state;
        }
        if (predicted_byte == data[i+1]) {
            correct++;
        }
    }
    return (double)correct / total;
}

// ---------- Сохранение генотипа в JSON ----------
void save_genotype(const std::vector<std::vector<int>>& genotype, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) return;
    file << "[\n";
    file << "  " << genotype[0][0] << ",\n";
    for (size_t i = 1; i < genotype.size(); ++i) {
        file << "  [";
        for (size_t j = 0; j < genotype[i].size(); ++j) {
            file << genotype[i][j];
            if (j != genotype[i].size() - 1) file << ", ";
        }
        file << "]";
        if (i != genotype.size() - 1) file << ",\n";
        else file << "\n";
    }
    file << "]\n";
    file.close();
}

// ---------- Главная функция ----------
int main() {
    try {
        // Загружаем данные
        std::vector<uint8_t> train_data = load_file(TRAIN_FILE);
        std::vector<uint8_t> valid_data = load_file(VALID_FILE);
        std::cout << "Train size: " << train_data.size() << " bytes\n";
        std::cout << "Valid size: " << valid_data.size() << " bytes\n";

        if (train_data.size() < 2 || valid_data.size() < 2) {
            std::cerr << "Data files too short" << std::endl;
            return 1;
        }

        // Структура для хранения результатов
        struct Candidate {
            std::vector<std::vector<int>> genotype;
            double train_acc;
            double valid_acc;
        };
        std::vector<Candidate> candidates;

        std::cout << "Starting random search...\n";
        for (int g = 0; g < NUM_GENOTYPES; ++g) {
            int num_neurons = MIN_NEURONS + (rand() % (MAX_NEURONS - MIN_NEURONS + 1));
            auto genotype = random_genotype(num_neurons);
            try {
                Network net = build_network_from_genotype(genotype);
                double acc = evaluate_prediction(net, train_data, MAX_STEPS, INPUT_NEURON);
                if (acc > 0.0) { // даже 0.4% – сохраняем, но лучше порог
                    candidates.push_back({genotype, acc, 0.0});
                    // Сортируем и оставляем только топ-100
                    std::sort(candidates.begin(), candidates.end(),
                              [](const Candidate& a, const Candidate& b) {
                                  return a.train_acc > b.train_acc;
                              });
                    if (candidates.size() > 100) candidates.pop_back();
                }
            } catch (const std::exception& e) {
                // игнорируем
            }
            if ((g+1) % 10000 == 0) {
                std::cout << "Tested " << (g+1) << ", best train acc so far: "
                          << (candidates.empty() ? 0.0 : candidates[0].train_acc) << std::endl;
            }
        }

        std::cout << "\nEvaluating top candidates on validation set...\n";
        for (auto& cand : candidates) {
            Network net = build_network_from_genotype(cand.genotype);
            cand.valid_acc = evaluate_prediction(net, valid_data, MAX_STEPS, INPUT_NEURON);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.valid_acc > b.valid_acc;
                  });

        std::cout << "\nTop 10 results:\n";
        for (size_t i = 0; i < std::min((size_t)10, candidates.size()); ++i) {
            std::cout << i+1 << ". train_acc=" << candidates[i].train_acc
                      << " valid_acc=" << candidates[i].valid_acc << std::endl;
            // Сохраняем лучшие генотипы
            std::string fname = "llm_best_" + std::to_string(i+1) + ".json";
            save_genotype(candidates[i].genotype, fname);
        }

        std::cout << "\nDone." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 
