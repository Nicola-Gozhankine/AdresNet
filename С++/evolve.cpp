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
const int NUM_GENOTYPES = 100000;        // сколько генотипов проверить
const int TEST_BYTES = 1024;             // 1 КБ
const int INPUT_NEURON = 0;
const int MAX_STEPS = 10;
const int NONCE_BITS = 16;
const int NONCE = 12345;
const int TOP_SAVE = 0;                  // 0 = сохранять все прошедшие, иначе только топ N по бонусам

// ---------- Генерация случайного генотипа (все в слое 0) ----------
std::vector<std::vector<int>> random_genotype_simple(int num_neurons) {
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
        // fixed0 и fixed1
        block[3] = 0;
        block[4] = local_dist(gen);
        block[5] = 0;
        block[6] = local_dist(gen);
        // адресные регистры (delta = 0)
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

// ---------- Генерация байтов ----------
void generate_bytes(Network& net, uint8_t* buffer, int num_bytes,
                    int input_neuron, int max_steps, int nonce_bits, int nonce) {
    int net_size = net.neurons.size();
    for (int i = 0; i < std::min(nonce_bits, net_size); ++i)
        net.neurons[i]->state = (nonce >> i) & 1;
    for (auto& n : net.neurons) n->inbox = 0;
    net._queue.clear(); net._in_queue.clear();

    int out_len = std::min(4, net_size);
    std::vector<int> output_neurons(out_len);
    for (int i = 0; i < out_len; ++i)
        output_neurons[i] = net_size - out_len + i;

    uint8_t byte = 0;
    int bitpos = 0, counter = 0, byte_idx = 0;
    int total_bits = num_bytes * 8;
    for (int b = 0; b < total_bits; ++b) {
        net.external_input(input_neuron, counter & 1);
        int steps = 0;
        while (!net.is_quiet() && steps < max_steps) { net.step(); ++steps; }
        int out = 0;
        for (int idx : output_neurons) out ^= net.neurons[idx]->state;
        byte = (byte << 1) | out;
        ++bitpos;
        if (bitpos == 8) { buffer[byte_idx++] = byte; byte = 0; bitpos = 0; }
        ++counter;
    }
}

// ---------- Базовые тесты (обязательные) ----------
bool monobit_test(const uint8_t* buffer, int num_bytes) {
    int ones = 0;
    int total_bits = num_bytes * 8;
    for (int i = 0; i < total_bits; ++i) {
        int byte_idx = i >> 3;
        int bit_pos = 7 - (i & 7);
        ones += (buffer[byte_idx] >> bit_pos) & 1;
    }
    int diff = std::abs(ones - total_bits/2);
    return diff <= total_bits * 0.05;   // 5% отклонение
}

bool runs_test(const uint8_t* buffer, int num_bytes) {
    int total_bits = num_bytes * 8;
    int max_run = 0, cur_run = 1;
    for (int i = 1; i < total_bits; ++i) {
        int prev_byte = (i-1)>>3, prev_bit = 7-((i-1)&7);
        int cur_byte = i>>3, cur_bit = 7-(i&7);
        int prev = (buffer[prev_byte]>>prev_bit)&1;
        int cur = (buffer[cur_byte]>>cur_bit)&1;
        if (cur == prev) cur_run++;
        else { if (cur_run > max_run) max_run = cur_run; cur_run = 1; }
    }
    if (cur_run > max_run) max_run = cur_run;
    return max_run <= 20;
}

bool autocorrelation_test(const uint8_t* buffer, int num_bytes, int lag, double tolerance=0.10) {
    int total_bits = num_bytes * 8;
    int same = 0;
    for (int i = 0; i < total_bits - lag; ++i) {
        int byte1 = i>>3, bit1 = 7-(i&7);
        int byte2 = (i+lag)>>3, bit2 = 7-((i+lag)&7);
        int b1 = (buffer[byte1]>>bit1)&1;
        int b2 = (buffer[byte2]>>bit2)&1;
        if (b1 == b2) same++;
    }
    int expected = (total_bits - lag) / 2;
    int diff = std::abs(same - expected);
    return diff <= expected * tolerance;
}

bool poker_test(const uint8_t* buffer, int num_bytes) {
    int nibble_counts[16] = {0};
    for (int i = 0; i < num_bytes; ++i) {
        uint8_t byte = buffer[i];
        nibble_counts[byte >> 4]++;
        nibble_counts[byte & 0x0F]++;
    }
    int total_nibbles = num_bytes * 2;
    int expected = total_nibbles / 16;
    int chi2 = 0;
    for (int i = 0; i < 16; ++i) {
        int diff = nibble_counts[i] - expected;
        chi2 += diff * diff;
    }
    return chi2 <= 2000;
}

// ---------- Бонусные тесты (дополнительные очки) ----------
bool runs_distribution_test(const uint8_t* buffer, int num_bytes) {
    // Проверяем распределение длин серий (1..10) через хи-квадрат
    int total_bits = num_bytes * 8;
    std::vector<int> observed(10, 0); // длины 1..10
    int cur_run = 1;
    for (int i = 1; i < total_bits; ++i) {
        int prev_byte = (i-1)>>3, prev_bit = 7-((i-1)&7);
        int cur_byte = i>>3, cur_bit = 7-(i&7);
        int prev = (buffer[prev_byte]>>prev_bit)&1;
        int cur = (buffer[cur_byte]>>cur_bit)&1;
        if (cur == prev) cur_run++;
        else {
            if (cur_run <= 10) observed[cur_run-1]++;
            cur_run = 1;
        }
    }
    if (cur_run <= 10) observed[cur_run-1]++;
    // Ожидаемые частоты для случайной последовательности (формула: expected_len = n / (2^(len+1)))
    double expected[10];
    for (int len = 1; len <= 10; ++len) {
        expected[len-1] = total_bits / (1 << (len+1));
    }
    double chi2 = 0;
    for (int i = 0; i < 10; ++i) {
        if (expected[i] > 0) {
            double diff = observed[i] - expected[i];
            chi2 += diff * diff / expected[i];
        }
    }
    // Порог: 16.919 для 9 степеней свободы (p=0.05)
    return chi2 <= 25.0; // чуть либеральнее
}

bool unique_32bit_words_test(const uint8_t* buffer, int num_bytes) {
    // Из первых 1 КБ можно извлечь 256 32-битных слов
    int num_words = num_bytes / 4;
    std::set<uint32_t> words;
    for (int i = 0; i < num_words; ++i) {
        uint32_t w = (buffer[i*4] << 24) | (buffer[i*4+1] << 16) | (buffer[i*4+2] << 8) | buffer[i*4+3];
        words.insert(w);
    }
    return words.size() >= 200; // минимум 200 уникальных из 256
}

bool byte_chi2_test(const uint8_t* buffer, int num_bytes) {
    int counts[256] = {0};
    for (int i = 0; i < num_bytes; ++i) {
        counts[buffer[i]]++;
    }
    int expected = num_bytes / 256;
    double chi2 = 0;
    for (int i = 0; i < 256; ++i) {
        double diff = counts[i] - expected;
        chi2 += diff * diff / expected;
    }
    // Для 255 степеней свободы, 99-й процентиль ~ 310
    return chi2 <= 400.0;
}

// ---------- Оценка генотипа: возвращает количество бонусов (0..4) или -1 если база не пройдена ----------
int evaluate_genotype(Network& net) {
    uint8_t buffer[TEST_BYTES];
    generate_bytes(net, buffer, TEST_BYTES, INPUT_NEURON, MAX_STEPS, NONCE_BITS, NONCE);

    // Базовые тесты (обязательные)
    if (!monobit_test(buffer, TEST_BYTES)) return -1;
    if (!runs_test(buffer, TEST_BYTES)) return -1;
    if (!autocorrelation_test(buffer, TEST_BYTES, 1)) return -1;
    if (!autocorrelation_test(buffer, TEST_BYTES, 8)) return -1;
    if (!poker_test(buffer, TEST_BYTES)) return -1;

    // Бонусы
    int bonus = 0;
    if (runs_distribution_test(buffer, TEST_BYTES)) bonus++;
    if (unique_32bit_words_test(buffer, TEST_BYTES)) bonus++;
    if (byte_chi2_test(buffer, TEST_BYTES)) bonus++;
    // четвёртый бонус – например, autocorrelation на сдвигах 16 и 32
    if (autocorrelation_test(buffer, TEST_BYTES, 16) && autocorrelation_test(buffer, TEST_BYTES, 32)) bonus++;

    return bonus;
}

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

int main() {
    std::cout << "Starting evolution with " << NUM_GENOTYPES << " random genotypes...\n";
    int saved_count = 0;
    for (int g = 0; g < NUM_GENOTYPES; ++g) {
        int num_neurons = 4 + (rand() % 27); // 4..30
        auto genotype = random_genotype_simple(num_neurons);
        try {
            Network net = build_network_from_genotype(genotype);
            int bonus = evaluate_genotype(net);
            if (bonus >= 0) {
                saved_count++;
                std::string fname = "Ген_крито_" + std::to_string(saved_count) + "_" + std::to_string(bonus) + ".json";
                save_genotype(genotype, fname);
                std::cout << "Found genotype #" << saved_count << " with " << num_neurons << " neurons, bonus=" << bonus << " saved to " << fname << std::endl;
            }
        } catch (const std::exception& e) {
            // игнорируем
        }
        if ((g+1) % 10000 == 0) {
            std::cout << "Tested " << (g+1) << " genotypes, saved " << saved_count << std::endl;
        }
    }
    std::cout << "Done. Total saved: " << saved_count << std::endl;
    return 0;
}