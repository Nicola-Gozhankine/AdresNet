#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <set>
#include "adresnet.cpp"

// Параметры тестирования
struct TestParams {
    int num_bytes;
    const char* label;
};

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

void compute_metrics(const uint8_t* buffer, int num_bytes, std::vector<double>& metrics) {
    int total_bits = num_bytes * 8;
    int ones = 0;
    for (int i = 0; i < total_bits; ++i) {
        int byte_idx = i >> 3, bit_pos = 7 - (i & 7);
        ones += (buffer[byte_idx] >> bit_pos) & 1;
    }
    double freq_dev = std::abs(ones - total_bits/2) / (double)total_bits;

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

    auto autocorr = [&](int lag) {
        int same = 0;
        for (int i = 0; i < total_bits - lag; ++i) {
            int byte1 = i>>3, bit1 = 7-(i&7);
            int byte2 = (i+lag)>>3, bit2 = 7-((i+lag)&7);
            if (((buffer[byte1]>>bit1)&1) == ((buffer[byte2]>>bit2)&1)) same++;
        }
        int expected = (total_bits - lag) / 2;
        return std::abs(same - expected) / (double)expected;
    };
    double corr1 = autocorr(1);
    double corr8 = autocorr(8);
    double corr16 = autocorr(16);

    int nibble_counts[16] = {0};
    for (int i = 0; i < num_bytes; ++i) {
        uint8_t b = buffer[i];
        nibble_counts[b>>4]++;
        nibble_counts[b&0x0F]++;
    }
    int total_nibbles = num_bytes * 2;
    int expected_nibble = total_nibbles / 16;
    int chi2_nibble = 0;
    for (int i = 0; i < 16; ++i) {
        int diff = nibble_counts[i] - expected_nibble;
        chi2_nibble += diff * diff;
    }

    int byte_counts[256] = {0};
    for (int i = 0; i < num_bytes; ++i) byte_counts[buffer[i]]++;
    int expected_byte = num_bytes / 256;
    double chi2_byte = 0;
    for (int i = 0; i < 256; ++i) {
        double diff = byte_counts[i] - expected_byte;
        chi2_byte += diff * diff / expected_byte;
    }

    int num_words = num_bytes / 4;
    std::set<uint32_t> words;
    for (int i = 0; i < num_words; ++i) {
        uint32_t w = (buffer[i*4]<<24)|(buffer[i*4+1]<<16)|(buffer[i*4+2]<<8)|buffer[i*4+3];
        words.insert(w);
    }
    int unique_words = words.size();

    metrics = {freq_dev, (double)max_run, corr1, corr8, corr16, (double)chi2_nibble, chi2_byte, (double)unique_words};
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <genotype.json> [max_steps=10] [nonce=12345]" << std::endl;
        return 1;
    }
    const char* genotype_file = argv[1];
    int max_steps = (argc > 2) ? std::stoi(argv[2]) : 10;
    int nonce = (argc > 3) ? std::stoi(argv[3]) : 12345;
    const int nonce_bits = 16;
    const int input_neuron = 0;

    // Объёмы для тестирования
    std::vector<TestParams> tests = {
        {1024, "1KB"},
        {102400, "100KB"},
        {1048576, "1MB"},
        {10485760, "10MB"}
    };

    try {
        auto genotype = JSONParser::parse(genotype_file);
        Network net = build_network_from_genotype(genotype);
        std::cout << "Testing genotype: " << genotype_file << std::endl;
        std::cout << "Neurons: " << net.neurons.size() << ", max_steps=" << max_steps << ", nonce=" << nonce << std::endl;
        std::cout << "Size\tFreqDev\tMaxRun\tCorr1\tCorr8\tCorr16\tChi2Nibble\tChi2Byte\tUnique32" << std::endl;

        for (auto& t : tests) {
            std::vector<uint8_t> buffer(t.num_bytes);
            generate_bytes(net, buffer.data(), t.num_bytes, input_neuron, max_steps, nonce_bits, nonce);
            std::vector<double> metrics;
            compute_metrics(buffer.data(), t.num_bytes, metrics);
            std::cout << t.label << "\t";
            for (double v : metrics) std::cout << v << "\t";
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 
