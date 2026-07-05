#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include "adresnet.cpp"   // Подключаем всю библиотеку

// ---------- Параметры ----------
const int NUM_MUTANTS_PER_TYPE = 500;   // сколько мутантов для каждого типа мутации
const int MAX_NEURONS = 128;            // для генерации случайных целей

// Загрузка генотипа из JSON
std::vector<std::vector<int>> load_genotype(const std::string& filename) {
    return JSONParser::parse(filename);
}

// Оценка точности для XOR (4 теста)
double evaluate_xor(Network& net) {
    int truth_table[4] = {0,1,1,0};
    int correct = 0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            net.reset();
            net.external_input(0, a);
            net.external_input(1, b);
            int steps = 0;
            while (!net.is_quiet() && steps < 10) { net.step(); ++steps; }
            int out = net.neurons.back()->state;
            if (out == truth_table[a*2+b]) correct++;
        }
    }
    return correct / 4.0;
}

// Мутация: изменение одного параметра в одном нейроне (фиксированный тип мутации)
void mutate_genotype(std::vector<std::vector<int>>& genotype, std::mt19937& rng, int mut_type) {
    int num_neurons = genotype[0][0];
    if (num_neurons == 0) return;
    std::uniform_int_distribution<int> neuron_dist(1, num_neurons);
    int neuron_idx = neuron_dist(rng);
    auto& block = genotype[neuron_idx];
    std::uniform_int_distribution<int> type_dist(0,1);
    std::uniform_int_distribution<int> mode_dist(0,1);
    std::uniform_int_distribution<int> layer_dist(0, 1000);
    std::uniform_int_distribution<int> local_dist(0, MAX_NEURONS-1);
    std::uniform_int_distribution<int> delta_dist(-1000, 1000);
    std::uniform_int_distribution<int> action_dist(0,4);
    switch (mut_type) {
        case 0: block[1] = type_dist(rng); break;
        case 1: block[2] = mode_dist(rng); break;
        case 2: block[3] = layer_dist(rng); break;
        case 3: block[4] = local_dist(rng); break;
        case 4: block[5] = layer_dist(rng); break;
        case 5: block[6] = local_dist(rng); break;
        case 6: block[7] = delta_dist(rng); break;
        case 7: block[8] = local_dist(rng); break;
        case 8: block[9] = delta_dist(rng); break;
        case 9: block[10] = local_dist(rng); break;
        case 10: block[11] = delta_dist(rng); break;
        case 11: block[12] = local_dist(rng); break;
        case 12: block[13] = delta_dist(rng); break;
        case 13: block[14] = local_dist(rng); break;
        case 14: block[15] = action_dist(rng); break;
        case 15: block[16] = action_dist(rng); break;
        case 16: block[17] = action_dist(rng); break;
        case 17: block[18] = action_dist(rng); break;
    }
}

int main() {
    // Загружаем лучший генотип для XOR (укажите правильный файл)
    std::string genotype_file = "best_xor_time.json";  // измените на существующий
    auto original = load_genotype(genotype_file);
    if (original.empty()) {
        std::cerr << "Failed to load " << genotype_file << std::endl;
        return 1;
    }
    std::cout << "Loaded genotype with " << original[0][0] << " neurons\n";
    
    // Строим сеть для исходного генотипа и оцениваем
    Network original_net = build_network_from_genotype(original);
    double original_acc = evaluate_xor(original_net);
    std::cout << "Original accuracy: " << original_acc << std::endl;
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Статистика по типам мутаций (0..17)
    struct Stats {
        int improve = 0;
        int worsen = 0;
        int same = 0;
        double sum_delta = 0.0;
    };
    std::vector<Stats> stats(18);
    
    for (int mut_type = 0; mut_type < 18; ++mut_type) {
        std::cout << "Testing mutation type " << mut_type << std::endl;
        for (int m = 0; m < NUM_MUTANTS_PER_TYPE; ++m) {
            auto mutant = original;
            mutate_genotype(mutant, rng, mut_type);
            try {
                Network net = build_network_from_genotype(mutant);
                double acc = evaluate_xor(net);
                double delta = acc - original_acc;
                if (delta > 1e-6) stats[mut_type].improve++;
                else if (delta < -1e-6) stats[mut_type].worsen++;
                else stats[mut_type].same++;
                stats[mut_type].sum_delta += delta;
            } catch (const std::exception& e) {
                // Если сеть не построилась, считаем ухудшением
                stats[mut_type].worsen++;
                stats[mut_type].sum_delta += (-original_acc);
            }
        }
    }
    
    // Вывод статистики
    std::cout << "\nMutation type statistics (improve/worsen/same/avg_delta)\n";
    for (int t = 0; t < 18; ++t) {
        int total = stats[t].improve + stats[t].worsen + stats[t].same;
        double avg_delta = stats[t].sum_delta / total;
        std::cout << t << ": " << stats[t].improve << "/" << stats[t].worsen << "/" << stats[t].same
                  << " avg_delta=" << avg_delta << std::endl;
    }
    
    return 0;
}