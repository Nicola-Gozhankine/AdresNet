// test_metrics.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include "adresnet.cpp"

const int TEST_BYTES = 1024;
const int INPUT_NEURON = 0;
const int MAX_STEPS = 10;
const int NONCE_BITS = 16;
const int NONCE = 12345;

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <genotype.json>" << std::endl;
        return 1;
    }
    try {
        auto genotype = JSONParser::parse(argv[1]);
        Network net = build_network_from_genotype(genotype);
        uint8_t buffer[TEST_BYTES];
        generate_bytes(net, buffer, TEST_BYTES, INPUT_NEURON, MAX_STEPS, NONCE_BITS, NONCE);
        
        int total_bits = TEST_BYTES * 8;
        int ones = 0;
        for (int i = 0; i < total_bits; ++i) {
            int byte_idx = i >> 3, bit_pos = 7 - (i & 7);
            ones += (buffer[byte_idx] >> bit_pos) & 1;
        }
        double freq_dev = std::abs(ones - total_bits/2) / (double)total_bits * 100.0;
        
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
        
        auto autocorr_dev = [&](int lag) -> double {
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
            return diff / (double)expected * 100.0;
        };
        double corr1_dev = autocorr_dev(1);
        double corr8_dev = autocorr_dev(8);
        
        int nibble_counts[16] = {0};
        for (int i = 0; i < TEST_BYTES; ++i) {
            uint8_t byte = buffer[i];
            nibble_counts[byte >> 4]++;
            nibble_counts[byte & 0x0F]++;
        }
        int total_nibbles = TEST_BYTES * 2;
        int expected = total_nibbles / 16;
        int chi2 = 0;
        for (int i = 0; i < 16; ++i) {
            int diff = nibble_counts[i] - expected;
            chi2 += diff * diff;
        }
        
        // Вывод в CSV: filename, freq_dev(%), max_run, corr1_dev(%), corr8_dev(%), chi2
        std::cout << argv[1] << "," << freq_dev << "," << max_run << "," << corr1_dev << "," << corr8_dev << "," << chi2 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 
