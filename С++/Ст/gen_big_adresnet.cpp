#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include "adresnet.cpp" // предполагается, что adresnet.cpp содержит все нужные классы и функции

// Функция для генерации большого потока байт и записи в бинарный файл
void generate_big_keystream(const char* genotype_file, long long num_bytes,
                            int input_neuron, int* output_neurons, int out_len,
                            int max_steps, int nonce_bits, int nonce,
                            const char* out_filename) {
    try {
        // Парсим генотип
        auto genotype = JSONParser::parse(genotype_file);
        auto net = build_network_from_genotype(genotype);
        
        // Инициализация состояний из nonce
        for (int i = 0; i < std::min(nonce_bits, (int)net.neurons.size()); ++i) {
            net.neurons[i]->state = (nonce >> i) & 1;
        }
        // Сброс буферов и очереди
        for (auto& n : net.neurons) n->inbox = 0;
        net._queue.clear();
        net._in_queue.clear();
        
        // Открываем выходной файл
        std::ofstream outfile(out_filename, std::ios::binary);
        if (!outfile) {
            throw std::runtime_error("Cannot open output file");
        }
        
        std::vector<uint8_t> buffer;
        buffer.reserve(4096); // буфер для байтов
        uint8_t byte = 0;
        int bitpos = 0;
        int counter = 0;
        long long total_bits = num_bytes * 8;
        
        std::cout << "Generating " << num_bytes << " bytes (" << total_bits << " bits)..." << std::endl;
        
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
            
            // Записываем буфер, когда накопится достаточно
            if (buffer.size() >= 4096) {
                outfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                buffer.clear();
            }
            
            // Прогресс
            if (b % (total_bits / 100) == 0) {
                std::cout << "\rProgress: " << (b * 100 / total_bits) << "%" << std::flush;
            }
        }
        // Остатки
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
    // Параметры (можно передавать через аргументы, но для простоты зададим здесь)
    const char* genotype_file = "good_gen_78847.json";
    long long num_bytes = 200 * 1024 ; // 200 МБ
    int input_neuron = 0;
    int output_neurons[] = {127, 126, 125, 124};
    int out_len = 4;
    int max_steps = 10;
    int nonce_bits = 16;
    int nonce = 12345;
    const char* out_filename = "stream_adresnet_2k.bin";
    
    std::cout << "Using genotype: " << genotype_file << std::endl;
    std::cout << "Output file: " << out_filename << std::endl;
    generate_big_keystream(genotype_file, num_bytes, input_neuron, output_neurons, out_len,
                           max_steps, nonce_bits, nonce, out_filename);
    return 0;
} 
