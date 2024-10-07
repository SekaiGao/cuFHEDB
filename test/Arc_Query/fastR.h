#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <array>
void writeToFile(const std::string& filename, 
                 const std::array<uint32_t, 1024>& sk,
                 const std::array<std::array<std::array<std::array<double, 1024UL>, 2UL>, 6UL>, 672UL>& bkfft,
                 const std::array<std::array<std::array<std::array<uint32_t, 673>, 1023>, 2>, 1024>& isk) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write `sk`
    outFile.write(reinterpret_cast<const char*>(sk.data()), sk.size() * sizeof(uint32_t));

    // Write `bkfft`
    for (const auto& layer1 : bkfft) {
        for (const auto& layer2 : layer1) {
            for (const auto& layer3 : layer2) {
            outFile.write(reinterpret_cast<const char*>(layer3.data()), layer3.size() * sizeof(double));
            }
        }
    }

    // Write `isk`
    for (const auto& layer1 : isk) {
        for (const auto& layer2 : layer1) {
            for (const auto& layer3 : layer2) {
                outFile.write(reinterpret_cast<const char*>(layer3.data()), layer3.size() * sizeof(uint32_t));
            }
        }
    }

    outFile.close();
}

void readFromFile(const std::string& filename, 
                  std::array<uint32_t, 1024>& sk,
                  std::array<std::array<std::array<std::array<double, 1024UL>, 2UL>, 6UL>, 672UL>& bkfft,
                  std::array<std::array<std::array<std::array<uint32_t, 673>, 1023>, 2>, 1024>& isk) {
    std::ifstream inFile(filename, std::ios::binary);

    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    // Read `sk`
    inFile.read(reinterpret_cast<char*>(sk.data()), sk.size() * sizeof(uint32_t));

    // Read `bkfft`
    for (auto& layer1 : bkfft) {
        for (auto& layer2 : layer1) {
          for (auto &layer3 : layer2) {
            inFile.read(reinterpret_cast<char*>(layer3.data()), layer3.size() * sizeof(double));
          }
        }
    }

    // Read `isk`
    for (auto& layer1 : isk) {
        for (auto& layer2 : layer1) {
            for (auto& layer3 : layer2) {
                inFile.read(reinterpret_cast<char*>(layer3.data()), layer3.size() * sizeof(uint32_t));
            }
        }
    }

    inFile.close();
}

std::string path = "/home/gaoshijie/cuHEDB/test/HE3_Query/key.bin";