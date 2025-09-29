// Logging.h
#pragma once
#include <fstream>
#include <string>

inline void logToFile(const std::string& msg) {
    static std::ofstream logFile(
        "C:/temp/juicer_debug.log", // writable path
        std::ios::out | std::ios::app
    );
    if (logFile.is_open()) {
        logFile << msg << std::endl;
    }
}
