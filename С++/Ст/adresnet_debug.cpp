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
#include <cmath>
#include <cctype>

// ---------- JSON парсер (улучшенный, с отладочными выводами) ----------
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

// ---------- Нейроны (полностью совпадают с Python-логикой) ----------
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

    virtual int step(int& target) = 0;

protected:
    int computeX() {
        if (mode == 0) {
            int x = inbox;
            inbox = 0;
            return x;
        } else {
            int x = inbox & 1;
            inbox = 0;
            return x;
        }
    }
};

class SimpleNeuron : public Neuron {
public:
    int target_gids[2];

    SimpleNeuron(int l, int s = 0, int m = 0) : Neuron(l, s, m) {
        target_gids[0] = target_gids[1] = -1;
    }

    int step(int& target) override {
        int x = computeX();
        int s_old = state;
        int y = x ^ s_old;
        state = 1 - s_old;
        target = target_gids[y];
        return y;
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

    int step(int& target) override {
        int x = computeX();
        int s_old = state;
        int y = x ^ s_old;
        state = 1 - s_old;
        int idx = (s_old << 1) | x;
        int act = action_list[idx];
        if (act == 0) target = fixed_gids[y];
        else target = address_gids[act-1];
        return y;
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
        int target = -1;
        int y = neurons[gid]->step(target);
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

// ---------- Построение сети из генотипа (с отладочным выводом) ----------
Network build_network_from_genotype(const std::vector<std::vector<int>>& genotype) {
    int num_neurons = genotype[0][0];
    std::cout << "Building network with " << num_neurons << " neurons" << std::endl;

    std::vector<int> neuron_types, neuron_modes, neuron_layers;
    std::vector<std::pair<int,int>> fixed0, fixed1;
    std::vector<std::vector<std::pair<int,int>>> addr;
    std::vector<std::vector<int>> actions;

    for (int i = 1; i <= num_neurons; ++i) {
        const auto& b = genotype[i];
        neuron_layers.push_back(b[0]);
        neuron_types.push_back(b[1]);
        neuron_modes.push_back(b[2]);
        fixed0.emplace_back(b[3], b[4]);
        fixed1.emplace_back(b[5], b[6]);
        std::vector<std::pair<int,int>> a;
        for (int j = 0; j < 4; ++j)
            a.emplace_back(b[7 + j*2], b[8 + j*2]);
        addr.push_back(a);
        std::vector<int> act;
        for (int j = 0; j < 4; ++j) act.push_back(b[15 + j]);
        actions.push_back(act);
    }

    Network net;
    for (int i = 0; i < num_neurons; ++i) {
        if (neuron_types[i] == 0) {
            net.add_neuron(std::make_unique<SimpleNeuron>(neuron_layers[i], 0, neuron_modes[i]));
        } else {
            net.add_neuron(std::make_unique<AddressableNeuron>(neuron_layers[i], 0, neuron_modes[i]));
        }
    }

    for (int gid = 0; gid < num_neurons; ++gid) {
        auto* neuron = net.neurons[gid].get();
        if (neuron_types[gid] == 0) {
            auto* simp = static_cast<SimpleNeuron*>(neuron);
            int t0 = net.local_to_global(fixed0[gid].first, fixed0[gid].second);
            int t1 = net.local_to_global(fixed1[gid].first, fixed1[gid].second);
            simp->target_gids[0] = (t0 == -1) ? 0 : t0;
            simp->target_gids[1] = (t1 == -1) ? 0 : t1;
            if (gid < 5) {
                std::cout << "Simple neuron " << gid << ": layer=" << neuron_layers[gid]
                          << ", mode=" << neuron_modes[gid]
                          << ", targets=[" << simp->target_gids[0] << "," << simp->target_gids[1] << "]"
                          << std::endl;
            }
        } else {
            auto* addr_neuron = static_cast<AddressableNeuron*>(neuron);
            int f0 = net.local_to_global(fixed0[gid].first, fixed0[gid].second);
            int f1 = net.local_to_global(fixed1[gid].first, fixed1[gid].second);
            addr_neuron->fixed_gids[0] = (f0 == -1) ? 0 : f0;
            addr_neuron->fixed_gids[1] = (f1 == -1) ? 0 : f1;
            for (int j = 0; j < 4; ++j) {
                int delta = addr[gid][j].first;
                int local = addr[gid][j].second;
                int target_layer = neuron_layers[gid] + delta;
                int gid_target = net.local_to_global(target_layer, local);
                addr_neuron->address_gids[j] = (gid_target == -1) ? 0 : gid_target;
            }
            for (int j = 0; j < 4; ++j)
                addr_neuron->action_list[j] = actions[gid][j];
            if (gid < 5) {
                std::cout << "Addressable neuron " << gid << ": layer=" << neuron_layers[gid]
                          << ", mode=" << neuron_modes[gid]
                          << ", fixed=[" << addr_neuron->fixed_gids[0] << "," << addr_neuron->fixed_gids[1] << "]"
                          << ", addr=[" << addr_neuron->address_gids[0] << "," << addr_neuron->address_gids[1]
                          << "," << addr_neuron->address_gids[2] << "," << addr_neuron->address_gids[3] << "]"
                          << ", actions=[" << addr_neuron->action_list[0] << "," << addr_neuron->action_list[1]
                          << "," << addr_neuron->action_list[2] << "," << addr_neuron->action_list[3] << "]"
                          << std::endl;
            }
        }
    }
    return net;
}

// ---------- Отладочная генерация нескольких битов (без создания библиотеки) ----------
void test_generation(Network& net, int input_neuron, const std::vector<int>& output_neurons,
                     int max_steps, int nonce_bits, int nonce, int num_bits) {
    // Инициализация состояния из nonce
    for (int i = 0; i < std::min(nonce_bits, (int)net.neurons.size()); ++i) {
        net.neurons[i]->state = (nonce >> i) & 1;
    }
    for (auto& n : net.neurons) n->inbox = 0;
    net._queue.clear();
    net._in_queue.clear();

    std::cout << "\nInitial states after nonce: ";
    for (int i = 0; i < std::min(10, (int)net.neurons.size()); ++i)
        std::cout << net.neurons[i]->state << " ";
    std::cout << std::endl;

    int counter = 0;
    for (int b = 0; b < num_bits; ++b) {
        std::cout << "\n--- Bit " << b << " (counter=" << counter << ") ---" << std::endl;
        net.external_input(input_neuron, counter & 1);
        std::cout << "Queue after input: ";
        for (int q : net._queue) std::cout << q << " ";
        std::cout << std::endl;

        int steps = 0;
        while (!net.is_quiet() && steps < max_steps) {
            net.step();
            ++steps;
            std::cout << "  Step " << steps << " done, queue now: ";
            for (int q : net._queue) std::cout << q << " ";
            std::cout << std::endl;
        }

        int out = 0;
        for (int idx : output_neurons) out ^= net.neurons[idx]->state;
        std::cout << "Output bits: ";
        for (int idx : output_neurons) std::cout << net.neurons[idx]->state << " ";
        std::cout << " -> XOR = " << out << std::endl;

        ++counter;
    }
}

int main() {
    try {
        std::string genotype_file = "good_genotype_75.json";
        std::cout << "Loading genotype from " << genotype_file << "..." << std::endl;
        auto genotype = JSONParser::parse(genotype_file);
        int num_neurons = genotype[0][0];
        std::cout << "Number of neurons: " << num_neurons << std::endl;

        // Параметры
        const int INPUT_NEURON = 0;
        const int MAX_STEPS = 10;
        const int NONCE_BITS = 16;
        const int NONCE = 12345;
        std::vector<int> output_neurons;
        for (int i = 0; i < 4; ++i) output_neurons.push_back(num_neurons - 1 - i);

        auto net = build_network_from_genotype(genotype);

        // Генерируем первые 5 битов для отладки
        test_generation(net, INPUT_NEURON, output_neurons, MAX_STEPS, NONCE_BITS, NONCE, 5);

        // Также выведем состояние после неполного сброса
        std::cout << "\nFinal states after test: ";
        for (int i = 0; i < std::min(10, (int)net.neurons.size()); ++i)
            std::cout << net.neurons[i]->state << " ";
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}