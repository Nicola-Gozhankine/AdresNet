// adresnet_lib.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <cstring>

// ---------- JSON парсер (минимальный) ----------
class JSONParser {
public:
    static std::vector<std::vector<int>> parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file");
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return parseArray(content);
    }

private:
    static std::vector<std::vector<int>> parseArray(const std::string& str) {
        std::vector<std::vector<int>> result;
        size_t pos = 0;
        while (pos < str.size() && str[pos] != '[') ++pos;
        if (pos == str.size()) throw std::runtime_error("Invalid JSON");
        ++pos;
        auto num = parseInt(str, pos);
        result.push_back({num});
        skipWhitespace(str, pos);
        if (str[pos] == ']') return result;
        if (str[pos] != ',') throw std::runtime_error("Expected ','");
        ++pos;
        while (true) {
            skipWhitespace(str, pos);
            if (str[pos] == ']') break;
            auto block = parseBlock(str, pos);
            result.push_back(block);
            skipWhitespace(str, pos);
            if (str[pos] == ']') break;
            if (str[pos] != ',') throw std::runtime_error("Expected ','");
            ++pos;
        }
        return result;
    }

    static std::vector<int> parseBlock(const std::string& str, size_t& pos) {
        if (str[pos] != '[') throw std::runtime_error("Expected '['");
        ++pos;
        std::vector<int> block;
        while (true) {
            skipWhitespace(str, pos);
            if (str[pos] == ']') break;
            int val = parseInt(str, pos);
            block.push_back(val);
            skipWhitespace(str, pos);
            if (str[pos] == ']') break;
            if (str[pos] != ',') throw std::runtime_error("Expected ','");
            ++pos;
        }
        ++pos;
        return block;
    }

    static int parseInt(const std::string& str, size_t& pos) {
        skipWhitespace(str, pos);
        int sign = 1;
        if (str[pos] == '-') { sign = -1; ++pos; }
        int val = 0;
        while (pos < str.size() && std::isdigit(str[pos])) {
            val = val * 10 + (str[pos] - '0');
            ++pos;
        }
        return sign * val;
    }

    static void skipWhitespace(const std::string& str, size_t& pos) {
        while (pos < str.size() && std::isspace(str[pos])) ++pos;
    }
};

// ---------- Нейроны ----------
class Neuron {
public:
    int layer;
    int local_id;
    int state;
    int mode;
    int inbox;

    Neuron(int l, int s = 0, int m = 0) : layer(l), local_id(-1), state(s), mode(m), inbox(0) {}
    virtual ~Neuron() {}

    void receive(int bit) {
        if (mode == 0) inbox |= bit;
        else if (bit) inbox++;
    }

    virtual std::pair<int,int> step() = 0; // (y, target)
};

class SimpleNeuron : public Neuron {
public:
    int target_gids[2];

    SimpleNeuron(int l, int s = 0, int m = 0) : Neuron(l, s, m) {
        target_gids[0] = target_gids[1] = -1;
    }

    std::pair<int,int> step() override {
        int x;
        if (mode == 0) {
            x = inbox;
            inbox = 0;
        } else {
            x = inbox & 1;
            inbox = 0;
        }
        int s_old = state;
        int y = x ^ s_old;
        state = 1 - s_old;
        return {y, target_gids[y]};
    }
};

class AddressableNeuron : public Neuron {
public:
    int fixed_gids[2];
    int address_gids[4];
    int action_list[4];   // 0=ordinary, 1..4=addr0..addr3

    AddressableNeuron(int l, int s = 0, int m = 0) : Neuron(l, s, m) {
        fixed_gids[0] = fixed_gids[1] = -1;
        std::fill(address_gids, address_gids+4, -1);
        std::fill(action_list, action_list+4, 0);
    }

    std::pair<int,int> step() override {
        int x;
        if (mode == 0) {
            x = inbox;
            inbox = 0;
        } else {
            x = inbox & 1;
            inbox = 0;
        }
        int s_old = state;
        int y = x ^ s_old;
        state = 1 - s_old;
        int idx = (s_old << 1) | x;
        int act = action_list[idx];
        if (act == 0) {
            return {y, fixed_gids[y]};
        } else {
            return {y, address_gids[act-1]};
        }
    }
};

// ---------- Сеть ----------
class Network {
public:
    std::vector<std::unique_ptr<Neuron>> neurons;
    std::unordered_map<int, std::unordered_map<int, int>> layer_to_global;
    std::deque<int> _queue;
    std::unordered_set<int> _in_queue;

    Network() = default;

    int add_neuron(std::unique_ptr<Neuron> neuron) {
        int gid = neurons.size();
        neurons.push_back(std::move(neuron));
        int layer = neurons.back()->layer;
        int local_id = layer_to_global[layer].size();
        layer_to_global[layer][local_id] = gid;
        neurons.back()->local_id = local_id;
        return gid;
    }

    int local_to_global(int layer, int local_id) const {
        auto it = layer_to_global.find(layer);
        if (it == layer_to_global.end()) return -1;
        auto jt = it->second.find(local_id);
        if (jt == it->second.end()) return -1;
        return jt->second;
    }

    void _enqueue(int gid) {
        if (_in_queue.find(gid) == _in_queue.end()) {
            _queue.push_back(gid);
            _in_queue.insert(gid);
        }
    }

    void external_input(int gid, int bit) {
        if (gid >= 0 && gid < (int)neurons.size()) {
            neurons[gid]->receive(bit);
            _enqueue(gid);
        }
    }

    bool step() {
        if (_queue.empty()) return false;
        int gid = _queue.front();
        _queue.pop_front();
        _in_queue.erase(gid);
        auto [y, target] = neurons[gid]->step();
        if (target >= 0 && target < (int)neurons.size()) {
            neurons[target]->receive(y);
            _enqueue(target);
        }
        return true;
    }

    void reset() {
        for (auto& n : neurons) {
            n->state = 0;
            n->inbox = 0;
        }
        _queue.clear();
        _in_queue.clear();
    }

    bool is_quiet() const { return _queue.empty(); }
};

// ---------- Построение сети из генотипа ----------
// ---------- Построение сети из генотипа ----------
Network build_network_from_genotype(const std::vector<std::vector<int>>& genotype) {
    int num_neurons = genotype[0][0];
    struct NeuronParams {
        int layer;
        int type;
        int mode;
        std::pair<int,int> fixed0;
        std::pair<int,int> fixed1;
        std::vector<std::pair<int,int>> addr;
        std::vector<int> actions;
    };
    std::vector<NeuronParams> params_list;
    for (int i = 1; i <= num_neurons; ++i) {
        const auto& b = genotype[i];
        NeuronParams p;
        p.layer = b[0];
        p.type = b[1];
        p.mode = b[2];
        p.fixed0 = {b[3], b[4]};
        p.fixed1 = {b[5], b[6]};
        for (int j = 0; j < 4; ++j) {
            p.addr.emplace_back(b[7 + j*2], b[8 + j*2]);
        }
        for (int j = 0; j < 4; ++j) {
            p.actions.push_back(b[15 + j]);
        }
        params_list.push_back(p);
    }

    Network net;
    for (const auto& p : params_list) {
        if (p.type == 0) {
            net.add_neuron(std::make_unique<SimpleNeuron>(p.layer, 0, p.mode));
        } else {
            net.add_neuron(std::make_unique<AddressableNeuron>(p.layer, 0, p.mode));
        }
    }

    for (int gid = 0; gid < num_neurons; ++gid) {
        auto* neuron = net.neurons[gid].get();
        const auto& p = params_list[gid];
        if (p.type == 0) {
            auto* simp = static_cast<SimpleNeuron*>(neuron);
            // fixed0 с коррекцией
            int layer0 = p.fixed0.first, local0 = p.fixed0.second;
            int t0;
            if (net.layer_to_global.find(layer0) != net.layer_to_global.end()) {
                int size0 = net.layer_to_global[layer0].size();
                if (local0 >= size0) local0 = local0 % size0;
                t0 = net.local_to_global(layer0, local0);
                if (t0 == -1) t0 = net.local_to_global(0,0);
            } else {
                t0 = net.local_to_global(0,0);
            }
            // fixed1 аналогично
            int layer1 = p.fixed1.first, local1 = p.fixed1.second;
            int t1;
            if (net.layer_to_global.find(layer1) != net.layer_to_global.end()) {
                int size1 = net.layer_to_global[layer1].size();
                if (local1 >= size1) local1 = local1 % size1;
                t1 = net.local_to_global(layer1, local1);
                if (t1 == -1) t1 = net.local_to_global(0,0);
            } else {
                t1 = net.local_to_global(0,0);
            }
            simp->target_gids[0] = t0;
            simp->target_gids[1] = t1;
        } else {
            auto* addr_neuron = static_cast<AddressableNeuron*>(neuron);
            // fixed цели с коррекцией
            int layer0 = p.fixed0.first, local0 = p.fixed0.second;
            int f0;
            if (net.layer_to_global.find(layer0) != net.layer_to_global.end()) {
                int size0 = net.layer_to_global[layer0].size();
                if (local0 >= size0) local0 = local0 % size0;
                f0 = net.local_to_global(layer0, local0);
                if (f0 == -1) f0 = net.local_to_global(0,0);
            } else {
                f0 = net.local_to_global(0,0);
            }
            int layer1 = p.fixed1.first, local1 = p.fixed1.second;
            int f1;
            if (net.layer_to_global.find(layer1) != net.layer_to_global.end()) {
                int size1 = net.layer_to_global[layer1].size();
                if (local1 >= size1) local1 = local1 % size1;
                f1 = net.local_to_global(layer1, local1);
                if (f1 == -1) f1 = net.local_to_global(0,0);
            } else {
                f1 = net.local_to_global(0,0);
            }
            addr_neuron->fixed_gids[0] = f0;
            addr_neuron->fixed_gids[1] = f1;

            // Адресные регистры
            for (int j = 0; j < 4; ++j) {
                int delta = p.addr[j].first;
                int local = p.addr[j].second;
                int target_layer = p.layer + delta;
                int gid_target;
                if (net.layer_to_global.find(target_layer) == net.layer_to_global.end()) {
                    gid_target = net.local_to_global(0,0);
                } else {
                    int size = net.layer_to_global[target_layer].size();
                    if (local >= size) {
                        local = local % size;
                    }
                    gid_target = net.local_to_global(target_layer, local);
                    if (gid_target == -1) gid_target = net.local_to_global(0,0);
                }
                addr_neuron->address_gids[j] = gid_target;
            }
            // Действия
            for (int j = 0; j < 4; ++j) {
                addr_neuron->action_list[j] = p.actions[j];
            }
        }
    }
    return net;
}










// ---------- Экспортируемые функции ----------
extern "C" {

// Генерирует ключевой поток по генотипу и параметрам
// Параметры:
//   genotype_file: путь к JSON-файлу генотипа
//   num_bytes: количество байт для генерации
//   input_neuron: индекс входного нейрона
//   output_neurons: массив индексов выходных нейронов
//   out_len: длина массива output_neurons
//   max_steps: максимальное число шагов обработки на один бит
//   nonce_bits: количество бит nonce для инициализации состояний
//   nonce: значение nonce
//   out_size: указатель, куда записывается размер результата
// Возвращает указатель на выделенный буфер (нужно освободить через free())
uint8_t* generate_keystream_from_file(const char* genotype_file, int num_bytes,
                                      int input_neuron, int* output_neurons, int out_len,
                                      int max_steps, int nonce_bits, int nonce,
                                      int* out_size) {
    try {
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

        // Генерация потока
        std::vector<uint8_t> keystream;
        keystream.reserve(num_bytes);
        uint8_t byte = 0;
        int bitpos = 0;
        int counter = 0;
        for (int b = 0; b < num_bytes * 8; ++b) {
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
                keystream.push_back(byte);
                byte = 0;
                bitpos = 0;
            }
            ++counter;
        }

        *out_size = keystream.size();
        uint8_t* buffer = (uint8_t*)malloc(keystream.size());
        memcpy(buffer, keystream.data(), keystream.size());
        return buffer;
    } catch (const std::exception& e) {
        *out_size = 0;
        return nullptr;
    }
}

} // extern "C"


extern "C" {

// Создаёт сеть из генотипа, возвращает указатель на объект Network*
void* create_network(const char* genotype_file) {
    try {
        auto genotype = JSONParser::parse(genotype_file);
        Network* net = new Network(build_network_from_genotype(genotype));
        return net;
    } catch (...) {
        return nullptr;
    }
}

// Уничтожает сеть
void destroy_network(void* net) {
    delete static_cast<Network*>(net);
}

// Сброс сети (состояния, буферы, очередь)
void reset_network(void* net) {
    static_cast<Network*>(net)->reset();
}

// Внешний вход
void external_input(void* net, int gid, int bit) {
    static_cast<Network*>(net)->external_input(gid, bit);
}

// Шаг сети
void step_network(void* net) {
    static_cast<Network*>(net)->step();
}

// Проверка, пуста ли очередь
int is_quiet(void* net) {
    return static_cast<Network*>(net)->is_quiet() ? 1 : 0;
}

// Получить состояние нейрона
int get_state(void* net, int gid) {
    return static_cast<Network*>(net)->neurons[gid]->state;
}

// Получить размер сети (количество нейронов)
int get_num_neurons(void* net) {
    return static_cast<Network*>(net)->neurons.size();
}











} // extern "C"

