#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "adresnet.cpp"

// ------------------------------------------------------------
//  Параметры эволюции
// ------------------------------------------------------------
const int INIT_NEURONS = 32;           // начальное число нейронов
const int POPULATION_SIZE = 100;       // мутантов на поколение
const int MAX_GENERATIONS = 200;       // максимум поколений
const int PATIENCE = 20;               // поколений без улучшения для остановки
const int MAX_STEPS = 25;              // шагов сети на бит
const int INPUT_NEURON = 0;            // входной нейрон
const double ADD_NEURON_PROB = 0.1;    // вероятность добавить нейрон при мутации
const double DEL_NEURON_PROB = 0.05;   // вероятность удалить нейрон

// ------------------------------------------------------------
//  Генерация синтетических данных (если нет файлов)
// ------------------------------------------------------------
void generate_synthetic_data() {
    // Базовый блок: 16 байт, детерминированный цикл
    std::vector<uint8_t> block = {
        0x41, 0x42, 0x43, 0x44, 0x42, 0x43, 0x44, 0x41,
        0x43, 0x44, 0x41, 0x42, 0x44, 0x41, 0x42, 0x43
    };
    // Обучающая последовательность: 64 повтора = 1024 байта
    std::ofstream train("train.bin", std::ios::binary);
    for (int i = 0; i < 64; ++i) {
        train.write(reinterpret_cast<const char*>(block.data()), block.size());
    }
    train.close();
    // Валидационная: 16 повторов, начиная с 4-го байта блока
    std::ofstream valid("valid.bin", std::ios::binary);
    for (int i = 0; i < 16; ++i) {
        valid.write(reinterpret_cast<const char*>(block.data() + 4), block.size() - 4);
        valid.write(reinterpret_cast<const char*>(block.data()), 4);
    }
    valid.close();
    std::cout << "Generated train.bin (1024 bytes) and valid.bin (256 bytes)\n";
}

// ------------------------------------------------------------
//  Загрузка данных
// ------------------------------------------------------------
std::vector<uint8_t> load_data(const std::string& fname) {
    std::ifstream f(fname, std::ios::binary);
    if (!f) {
        generate_synthetic_data();
        f.open(fname, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot create data file");
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

// ------------------------------------------------------------
//  Генерация случайного генотипа (слой 0, заданное число нейронов)
// ------------------------------------------------------------
std::vector<std::vector<int>> random_genotype(int num_neurons) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> type_dist(0,1);
    std::uniform_int_distribution<int> mode_dist(0,1);
    std::uniform_int_distribution<int> action_dist(0,4);
    std::uniform_int_distribution<int> local_dist(0, num_neurons-1);

    std::vector<std::vector<int>> genotype;
    genotype.push_back({num_neurons});
    for (int i = 0; i < num_neurons; ++i) {
        std::vector<int> block(19, 0);
        block[0] = 0;
        block[1] = type_dist(gen);
        block[2] = mode_dist(gen);
        block[3] = 0; block[4] = local_dist(gen);
        block[5] = 0; block[6] = local_dist(gen);
        for (int j = 0; j < 4; ++j) {
            block[7 + 2*j] = 0;
            block[8 + 2*j] = local_dist(gen);
        }
        for (int j = 0; j < 4; ++j) block[15 + j] = action_dist(gen);
        genotype.push_back(block);
    }
    return genotype;
}

// ------------------------------------------------------------
//  Сохранение генотипа в JSON
// ------------------------------------------------------------
void save_genotype(const std::vector<std::vector<int>>& g, const std::string& fname) {
    std::ofstream file(fname);
    if (!file) return;
    file << "[\n";
    file << "  " << g[0][0] << ",\n";
    for (size_t i = 1; i < g.size(); ++i) {
        file << "  [";
        for (size_t j = 0; j < g[i].size(); ++j) {
            file << g[i][j];
            if (j+1 < g[i].size()) file << ", ";
        }
        file << "]";
        if (i+1 < g.size()) file << ",\n";
        else file << "\n";
    }
    file << "]\n";
}

// ------------------------------------------------------------
//  Оценка точности предсказания следующего байта
// ------------------------------------------------------------
double evaluate(Network& net, const std::vector<uint8_t>& data, int max_steps, int input_neuron) {
    net.reset();
    int sz = net.neurons.size();
    if (sz < 8) return 0.0;
    std::vector<int> out(8);
    for (int i = 0; i < 8; ++i) out[i] = sz - 8 + i;
    int correct = 0;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        uint8_t cur = data[i];
        for (int bit = 7; bit >= 0; --bit) {
            net.external_input(input_neuron, (cur >> bit) & 1);
            for (int s = 0; s < max_steps; ++s) net.step();
        }
        uint8_t pred = 0;
        for (int b = 0; b < 8; ++b) pred = (pred << 1) | net.neurons[out[b]]->state;
        if (pred == data[i+1]) correct++;
    }
    return double(correct) / (data.size() - 1);
}

// ------------------------------------------------------------
//  Мутации (все 18 типов, равновероятно)
// ------------------------------------------------------------
void mutate(std::vector<std::vector<int>>& g, std::mt19937& rng) {
    int n = g[0][0];
    if (n == 0) return;
    std::uniform_int_distribution<int> neur(1, n);
    int idx = neur(rng);
    auto& blk = g[idx];
    std::uniform_int_distribution<int> mtype(0, 17);
    int t = mtype(rng);
    std::uniform_int_distribution<int> bit01(0,1);
    std::uniform_int_distribution<int> layer(0, 1000);
    std::uniform_int_distribution<int> local(0, 1000); // большой запас, потом скорректируется
    std::uniform_int_distribution<int> delta(-1000, 1000);
    std::uniform_int_distribution<int> act(0,4);
    switch(t) {
        case 0: blk[1] = bit01(rng); break;
        case 1: blk[2] = bit01(rng); break;
        case 2: blk[3] = layer(rng); break;
        case 3: blk[4] = local(rng); break;
        case 4: blk[5] = layer(rng); break;
        case 5: blk[6] = local(rng); break;
        case 6: blk[7] = delta(rng); break;
        case 7: blk[8] = local(rng); break;
        case 8: blk[9] = delta(rng); break;
        case 9: blk[10] = local(rng); break;
        case 10: blk[11] = delta(rng); break;
        case 11: blk[12] = local(rng); break;
        case 12: blk[13] = delta(rng); break;
        case 13: blk[14] = local(rng); break;
        case 14: blk[15] = act(rng); break;
        case 15: blk[16] = act(rng); break;
        case 16: blk[17] = act(rng); break;
        case 17: blk[18] = act(rng); break;
    }
}

// ------------------------------------------------------------
//  Добавление нового нейрона (слой 0, случайные связи)
// ------------------------------------------------------------
void add_neuron(std::vector<std::vector<int>>& g, std::mt19937& rng) {
    int old = g[0][0];
    g[0][0] = old + 1;
    std::vector<int> blk(19, 0);
    blk[0] = 0;
    std::uniform_int_distribution<int> bit01(0,1);
    std::uniform_int_distribution<int> local(0, old); // может ссылаться на существующих
    std::uniform_int_distribution<int> act(0,4);
    blk[1] = bit01(rng);
    blk[2] = bit01(rng);
    blk[3] = 0; blk[4] = local(rng);
    blk[5] = 0; blk[6] = local(rng);
    for (int j = 0; j < 4; ++j) {
        blk[7 + 2*j] = 0;
        blk[8 + 2*j] = local(rng);
    }
    for (int j = 0; j < 4; ++j) blk[15 + j] = act(rng);
    g.push_back(blk);
}

// ------------------------------------------------------------
//  Удаление нейрона (нельзя удалять последние 8, так как они выходные)
// ------------------------------------------------------------
void del_neuron(std::vector<std::vector<int>>& g, std::mt19937& rng) {
    int n = g[0][0];
    if (n <= 8) return; // не удаляем, если останется меньше 8 нейронов
    std::uniform_int_distribution<int> idx_dist(1, n); // 1..n
    int idx = idx_dist(rng);
    // Не удаляем выходные (последние 8)
    if (idx > n - 8) return;
    g.erase(g.begin() + idx);
    g[0][0] = n - 1;
    // Здесь нужно было бы пересчитать локальные ID в генотипе, но при сборке сети build_network_from_genotype
    // сама пересчитывает слои и корректирует адреса. Поэтому просто удаляем нейрон.
}

// ------------------------------------------------------------
//  Основная эволюция
// ------------------------------------------------------------
int main() {
    // Данные
    auto train = load_data("train.bin");
    auto valid = load_data("valid.bin");
    std::cout << "Train size: " << train.size() << " bytes\n";
    std::cout << "Valid size: " << valid.size() << " bytes\n";

    // Начальный генотип
    auto best = random_genotype(INIT_NEURONS);
    Network net = build_network_from_genotype(best);
    double best_acc = evaluate(net, valid, MAX_STEPS, INPUT_NEURON);
    std::cout << "Initial valid accuracy: " << best_acc << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> prob(0,1);

    int no_improve = 0;
    for (int gen = 1; gen <= MAX_GENERATIONS && no_improve < PATIENCE; ++gen) {
        std::vector<std::pair<double, std::vector<std::vector<int>>>> candidates;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            auto ind = best;
            // 1-2 мутации
            int num_mut = 1 + (prob(rng) < 0.5 ? 1 : 0);
            for (int m = 0; m < num_mut; ++m) {
                mutate(ind, rng);
                if (prob(rng) < ADD_NEURON_PROB) add_neuron(ind, rng);
                if (prob(rng) < DEL_NEURON_PROB) del_neuron(ind, rng);
            }
            try {
                Network n = build_network_from_genotype(ind);
                double acc = evaluate(n, valid, MAX_STEPS, INPUT_NEURON);
                candidates.emplace_back(acc, std::move(ind));
            } catch (const std::exception& e) {
                // игнорируем неудачные сети
            }
        }
        if (candidates.empty()) continue;
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        double new_acc = candidates[0].first;
        if (new_acc > best_acc) {
            best_acc = new_acc;
            best = std::move(candidates[0].second);
            no_improve = 0;
            save_genotype(best, "best_systematic.json");
            std::cout << "Gen " << gen << ": new best valid acc = " << best_acc
                      << " (neurons=" << best[0][0] << ")\n";
        } else {
            no_improve++;
            if (gen % 10 == 0) {
                std::cout << "Gen " << gen << ": no improvement, best = " << best_acc << "\n";
            }
        }
    }

    std::cout << "\nFinal valid accuracy: " << best_acc << std::endl;
    save_genotype(best, "best_systematic_final.json");
    std::cout << "Saved final genotype to best_systematic_final.json\n";

    // Дополнительно проверим на test.bin, если он есть
    try {
        auto test = load_data("test.bin");
        Network net_final = build_network_from_genotype(best);
        double test_acc = evaluate(net_final, test, MAX_STEPS, INPUT_NEURON);
        std::cout << "Test accuracy: " << test_acc << std::endl;
    } catch (...) {
        std::cout << "No test.bin, skipping test evaluation.\n";
    }
    return 0;
} 
