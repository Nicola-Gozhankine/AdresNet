// bm_lib.cpp
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>   // для FILE, fopen, fread, fclose

extern "C" {

int berlekamp_massey(const uint8_t* bits, int n) {
    std::vector<int> c(n+1, 0);
    std::vector<int> b(n+1, 0);
    std::vector<int> t(n+1, 0);
    c[0] = b[0] = 1;
    int L = 0;
    int m = -1;
    for (int N = 0; N < n; ++N) {
        int d = bits[N];
        for (int i = 1; i <= L; ++i)
            d ^= c[i] & bits[N-i];
        if (d) {
            t = c;
            int shift = N - m;
            for (int i = 0; i + shift <= n; ++i)
                c[i+shift] ^= b[i];
            if (L <= N/2) {
                L = N + 1 - L;
                m = N;
                b = t;
            }
        }
    }
    return L;
}

int compute_linear_complexity(const char* filename, int max_bits, int* result) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    std::vector<uint8_t> bits;
    bits.reserve(max_bits);
    unsigned char byte;
    while (bits.size() < (size_t)max_bits && fread(&byte, 1, 1, f) == 1) {
        for (int shift = 7; shift >= 0; --shift) {
            bits.push_back((byte >> shift) & 1);
            if (bits.size() == (size_t)max_bits) break;
        }
    }
    fclose(f);
    if (bits.empty()) return -1;
    *result = berlekamp_massey(bits.data(), bits.size());
    return 0;
}

} // extern "C"