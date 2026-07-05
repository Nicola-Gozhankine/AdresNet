#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Базовый блок: 16 байт
    std::vector<uint8_t> block = {
        0x41, 0x42, 0x43, 0x44, 0x42, 0x43, 0x44, 0x41,
        0x43, 0x44, 0x41, 0x42, 0x44, 0x41, 0x42, 0x43
    };
    // Обучающая последовательность: 12 повторов = 192 байта
    std::ofstream train("train.bin", std::ios::binary);
    for (int i = 0; i < 12; ++i) {
        train.write(reinterpret_cast<const char*>(block.data()), block.size());
    }
    train.close();

    // Валидационная: начинаем с 4-го байта блока (0x44) и берём 4 блока = 64 байта
    std::ofstream valid("valid.bin", std::ios::binary);
    for (int i = 0; i < 4; ++i) {
        valid.write(reinterpret_cast<const char*>(block.data() + 4), block.size() - 4);
        valid.write(reinterpret_cast<const char*>(block.data()), 4);
    }
    valid.close();
    std::cout << "Generated train.bin (" << 12*16 << " bytes) and valid.bin (" << 4*16 << " bytes)\n";
    return 0;
} 
