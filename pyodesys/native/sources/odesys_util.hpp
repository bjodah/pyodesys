#pragma once

#include <string>  // was not properly included in anyode.hpp
#include <unordered_map>  // was not properly included in anyode.hpp
#include <utility>

namespace odesys_util {
    // Helper for Cython wrapper
    std::pair<std::pair<std::vector<double>, std::vector<double> >, std::unordered_map<std::string, int> >
    adaptive_return(std::pair<std::vector<double>, std::vector<double> > xout_yout,
                    std::unordered_map<std::string, int> nfo){
        return std::make_pair(xout_yout, nfo);
    }
}
