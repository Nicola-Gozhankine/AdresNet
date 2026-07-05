#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include "adresnet.cpp"

// ---------- Параметры ----------
const int POPULATION_SIZE = 200;
const int GENERATIONS = 50;
const int MUTATIONS_PER_INDIVIDUAL = 2;
const double ADD_NEURON_PROB = 0.05;
const int MAX_NEURONS = 128;
const int MAX_STEPS = 25;
const int INPUT_NEURON = 0;

// ---------- Генерация случайного генотипа (слой 0) ----------
std::vector<std::vector<int>> random_genotype(int num_neurons) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> type_dist(0,1);
    std::uniform_int_distribution<int> mode_dist(0,1);
    std::uniform_int_distribution<int> action_dist(0,4);
    std::uniform_int_distribution<int> local_dist(0, num_neurons-1);

    std::vector<std::vector<int>> genotype;
    genotype.push_back({num_neurons});
    for (int i=0; i<num_neurons; ++i) {
        std::vector<int> block(19,0);
        block[0] = 0;
        block[1] = type_dist(gen);
        block[2] = mode_dist(gen);
        block[3] = 0; block[4] = local_dist(gen);
        block[5] = 0; block[6] = local_dist(gen);
        for (int j=0; j<4; ++j) {
            block[7+2*j] = 0;
            block[8+2*j] = local_dist(gen);
        }
        for (int j=0; j<4; ++j) block[15+j] = action_dist(gen);
        genotype.push_back(block);
    }
    return genotype;
}

// ---------- Загрузка/сохранение ----------
std::vector<std::vector<int>> load_genotype(const std::string& fname) {
    return JSONParser::parse(fname);
}
void save_genotype(const std::vector<std::vector<int>>& g, const std::string& fname) {
    std::ofstream file(fname);
    if (!file) return;
    file << "[\n";
    file << "  " << g[0][0] << ",\n";
    for (size_t i=1; i<g.size(); ++i) {
        file << "  [";
        for (size_t j=0; j<g[i].size(); ++j) {
            file << g[i][j];
            if (j+1<g[i].size()) file << ", ";
        }
        file << "]";
        if (i+1<g.size()) file << ",\n";
        else file << "\n";
    }
    file << "]\n";
}

// ---------- Оценка на синтетических данных ----------
double evaluate(Network& net, const std::vector<uint8_t>& data) {
    net.reset();
    int sz = net.neurons.size();
    if (sz < 8) return 0.0;
    std::vector<int> out(8);
    for (int i=0; i<8; ++i) out[i] = sz - 8 + i;
    int correct = 0;
    for (size_t i=0; i<data.size()-1; ++i) {
        uint8_t cur = data[i];
        for (int bit=7; bit>=0; --bit) {
            net.external_input(INPUT_NEURON, (cur>>bit)&1);
            for (int s=0; s<MAX_STEPS; ++s) net.step();
        }
        uint8_t pred = 0;
        for (int b=0; b<8; ++b) pred = (pred<<1) | net.neurons[out[b]]->state;
        if (pred == data[i+1]) correct++;
    }
    return double(correct) / (data.size()-1);
}

// ---------- Мутации (все типы) ----------
void mutate(std::vector<std::vector<int>>& g, std::mt19937& rng) {
    int n = g[0][0];
    if (n==0) return;
    std::uniform_int_distribution<int> neur(1, n);
    int idx = neur(rng);
    auto& blk = g[idx];
    std::uniform_int_distribution<int> mtype(0,17);
    int t = mtype(rng);
    std::uniform_int_distribution<int> bit01(0,1);
    std::uniform_int_distribution<int> layer(0,1000);
    std::uniform_int_distribution<int> local(0, MAX_NEURONS-1);
    std::uniform_int_distribution<int> delta(-1000,1000);
    std::uniform_int_distribution<int> act(0,4);
    switch(t) {
        case 0: blk[1] = bit01(rng); break;
        case 1: blk[2] = bit01(rng); break;
        case 2: blk[3] = layer(rng); break;
        case 3: blk[4] = local(rng); break;
        case 4: blk[5] = layer(rng); break;
        case 5: blk[6] = local(rng); break;
        case 6: blk[7] = delta(rng); break;
        case 7: blk[8] = local(rng); break;
        case 8: blk[9] = delta(rng); break;
        case 9: blk[10] = local(rng); break;
        case 10: blk[11] = delta(rng); break;
        case 11: blk[12] = local(rng); break;
        case 12: blk[13] = delta(rng); break;
        case 13: blk[14] = local(rng); break;
        case 14: blk[15] = act(rng); break;
        case 15: blk[16] = act(rng); break;
        case 16: blk[17] = act(rng); break;
        case 17: blk[18] = act(rng); break;
    }
}

void add_neuron(std::vector<std::vector<int>>& g, std::mt19937& rng) {
    int old = g[0][0];
    g[0][0] = old+1;
    std::vector<int> blk(19,0);
    blk[0]=0;
    std::uniform_int_distribution<int> bit01(0,1);
    std::uniform_int_distribution<int> local(0, old);
    std::uniform_int_distribution<int> act(0,4);
    blk[1]=bit01(rng); blk[2]=bit01(rng);
    blk[3]=0; blk[4]=local(rng);
    blk[5]=0; blk[6]=local(rng);
    for (int j=0;j<4;++j) { blk[7+2*j]=0; blk[8+2*j]=local(rng); }
    for (int j=0;j<4;++j) blk[15+j]=act(rng);
    g.push_back(blk);
}

int main() {
    std::ifstream train_f("train.bin", std::ios::binary);
    if (!train_f) { std::cerr << "train.bin not found\n"; return 1; }
    std::vector<uint8_t> train((std::istreambuf_iterator<char>(train_f)), {});
    std::cout << "Train size: " << train.size() << " bytes\n";

    // Загружаем лучший известный генотип (или создаём случайный)
    std::vector<std::vector<int>> best;
    try {
        best = load_genotype("llm_best_current.json");
    } catch(...) {
        best = random_genotype(21);
        save_genotype(best, "llm_best_current.json");
    }
    std::cout << "Initial neurons: " << best[0][0] << "\n";

    Network net = build_network_from_genotype(best);
    double best_acc = evaluate(net, train);
    std::cout << "Initial accuracy: " << best_acc << "\n";

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> add_prob(0,1);

    for (int gen=0; gen<GENERATIONS; ++gen) {
        std::vector<std::pair<double, std::vector<std::vector<int>>>> pop;
        for (int i=0; i<POPULATION_SIZE; ++i) {
            auto ind = best;
            for (int m=0; m<MUTATIONS_PER_INDIVIDUAL; ++m) {
                mutate(ind, rng);
                if (add_prob(rng) < ADD_NEURON_PROB) add_neuron(ind, rng);
            }
            try {
                Network n = build_network_from_genotype(ind);
                double acc = evaluate(n, train);
                pop.emplace_back(acc, std::move(ind));
            } catch(...) {}
        }
        std::sort(pop.begin(), pop.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        if (!pop.empty() && pop[0].first > best_acc) {
            best_acc = pop[0].first;
            best = std::move(pop[0].second);
            save_genotype(best, "llm_best_current.json");
            std::cout << "Gen " << gen+1 << ": new best " << best_acc << " (neurons=" << best[0][0] << ")\n";
        } else {
            std::cout << "Gen " << gen+1 << ": no improvement, best " << best_acc << "\n";
        }
    }
    save_genotype(best, "llm_best_final.json");
    std::cout << "Final accuracy: " << best_acc << "\n";
    return 0;
}