#include <iostream>
#include <fstream>
#include <vector>
#include "adresnet.cpp"

int main() {
    // Загружаем лучший генотип
    auto genotype = JSONParser::parse("llm_best_final.json");
    Network net = build_network_from_genotype(genotype);

    // Загружаем valid.bin
    std::ifstream f("valid.bin", std::ios::binary);
    if (!f) { std::cerr << "valid.bin not found\n"; return 1; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});

    const int MAX_STEPS = 25;
    const int INPUT_NEURON = 0;
    net.reset();

    int sz = net.neurons.size();
    if (sz < 8) { std::cerr << "Too few neurons\n"; return 1; }
    std::vector<int> out(8);
    for (int i = 0; i < 8; ++i) out[i] = sz - 8 + i;

    int correct = 0;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        uint8_t cur = data[i];
        // Подаём 8 бит текущего байта
        for (int bit = 7; bit >= 0; --bit) {
            net.external_input(INPUT_NEURON, (cur >> bit) & 1);
            for (int s = 0; s < MAX_STEPS; ++s) net.step();
        }
        // Считываем предсказанный байт
        uint8_t pred = 0;
        for (int b = 0; b < 8; ++b) {
            pred = (pred << 1) | net.neurons[out[b]]->state;
        }
        if (pred == data[i + 1]) correct++;
    }

    double acc = double(correct) / (data.size() - 1);
    std::cout << "Validation accuracy: " << acc << std::endl;
    return 0;
}