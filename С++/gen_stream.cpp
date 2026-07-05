#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include "adresnet.cpp"

void generate_keystream(const char* genotype_file, long long num_bytes,
                        int input_neuron, int max_steps, int nonce_bits, int nonce,
                        const char* out_filename) {
    try {
        auto genotype = JSONParser::parse(genotype_file);
        auto net = build_network_from_genotype(genotype);
        
        int net_size = net.neurons.size();
        if (net_size < 4) {
            std::cerr << "Error: network has less than 4 neurons, cannot generate output\n";
            return;
        }
        
        // Инициализация nonce
        for (int i = 0; i < std::min(nonce_bits, net_size); ++i) {
            net.neurons[i]->state = (nonce >> i) & 1;
        }
        for (auto& n : net.neurons) n->inbox = 0;
        net._queue.clear();
        net._in_queue.clear();
        
        // Выходные нейроны – последние 4
        int out_len = 4;
        int output_neurons[4];
        for (int i = 0; i < out_len; ++i) {
            output_neurons[i] = net_size - out_len + i;
        }
        
        std::ofstream outfile(out_filename, std::ios::binary);
        if (!outfile) throw std::runtime_error("Cannot open output file");
        
        std::vector<uint8_t> buffer;
        buffer.reserve(4096);
        uint8_t byte = 0;
        int bitpos = 0;
        int counter = 0;
        long long total_bits = num_bytes * 8;
        
        std::cout << "Generating " << num_bytes << " bytes (" << total_bits << " bits)..." << std::endl;
        std::cout << "Network size: " << net_size << " neurons, output neurons: ";
        for (int i = 0; i < out_len; ++i) std::cout << output_neurons[i] << " ";
        std::cout << std::endl;
        
        for (long long b = 0; b < total_bits; ++b) {
            net.external_input(input_neuron, counter & 1);
            int steps = 0;
            while (!net.is_quiet() && steps < max_steps) {
                net.step();
                ++steps;
            }
            int out = 0;
            for (int i = 0; i < out_len; ++i) {
                out ^= net.neurons[output_neurons[i]]->state;
            }
            byte = (byte << 1) | out;
            ++bitpos;
            if (bitpos == 8) {
                buffer.push_back(byte);
                byte = 0;
                bitpos = 0;
            }
            ++counter;
            
            if (buffer.size() >= 4096) {
                outfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                buffer.clear();
            }
            
            if (b % (total_bits / 100) == 0) {
                std::cout << "\rProgress: " << (b * 100 / total_bits) << "%" << std::flush;
            }
        }
        if (bitpos > 0) {
            byte <<= (8 - bitpos);
            buffer.push_back(byte);
        }
        if (!buffer.empty()) {
            outfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        }
        outfile.close();
        std::cout << "\nDone." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <genotype.json> <num_bytes> <out.bin> [input_neuron=0] [max_steps=10] [nonce_bits=16] [nonce=12345]" << std::endl;
        return 1;
    }
    const char* genotype_file = argv[1];
    long long num_bytes = std::stoll(argv[2]);
    const char* out_filename = argv[3];
    int input_neuron = (argc > 4) ? std::stoi(argv[4]) : 0;
    int max_steps = (argc > 5) ? std::stoi(argv[5]) : 10;
    int nonce_bits = (argc > 6) ? std::stoi(argv[6]) : 16;
    int nonce = (argc > 7) ? std::stoi(argv[7]) : 12345;
    
    std::cout << "Genotype: " << genotype_file << std::endl;
    std::cout << "Output: " << out_filename << " (" << num_bytes << " bytes)" << std::endl;
    generate_keystream(genotype_file, num_bytes, input_neuron, max_steps, nonce_bits, nonce, out_filename);
    return 0;
} 
