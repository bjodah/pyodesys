#include <cstdlib>
#include <sysexits.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <cvodes_anyode_parallel.hpp>
#include "odesys_anyode.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[]){
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("atol", po::value<realtype>()->default_value(1e-8), "absolute tolerance")
        ("rtol", po::value<realtype>()->default_value(1e-8), "relative tolerance")
        ("mxsteps", po::value<int>()->default_value(500), "maximum number of steps")
        ("lmm", po::value<std::string>()->default_value("BDF"), "linear multistep method")
        ("return_on_root", po::value<bool>()->default_value(false), "Return on root")
        ("autorestart", po::value<int>()->default_value(0), "Autorestart (autonomous)")
        ("return_on_error", po::value<bool>()->default_value(false), "Return on error")
        ("get_dx_max_factor", po::value<realtype>()->default_value(1.0), "get_dx_max multiplicative factor")
        ("error_outside_bounds", po::value<bool>()->default_value(false), "Return recoverable error to solver when outside bounds")
        ("special_settings", po::value<std::string>()->default_value(""), "special settings2 (if customized)")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return EX_USAGE;
    }

    const int ny = ${p_odesys.ny};
    const int nparams = ${len(p_odesys.params)};
    std::vector<odesys_anyode::OdeSys *> systems;
    const std::vector<realtype> atol(ny, vm["atol"].as<realtype>());
    const realtype rtol(vm["rtol"].as<realtype>());
    const auto lmm((vm["lmm"].as<std::string>() == "adams") ?
                   cvodes_cxx::LMM::Adams :
                   cvodes_cxx::LMM::BDF);

    std::vector<realtype> y0;

    std::vector<realtype> tout;
    int nout=-1;
    const long int mxsteps(vm["mxsteps"].as<int>());

    std::vector<realtype> dx0;
    std::vector<realtype> dx_min;
    std::vector<realtype> dx_max;
    const bool with_jacobian = true;
    auto iter_type=cvodes_cxx::IterType::Undecided;
    int linear_solver=0;
    const int maxl=0;
    const realtype eps_lin=0.0;
    const unsigned nderiv=0;
    bool return_on_root(vm["return_on_root"].as<bool>());
    int autorestart(vm["autorestart"].as<int>());
    bool return_on_error(vm["return_on_error"].as<bool>());

    std::vector<realtype> params;
    const realtype get_dx_max_factor(vm["get_dx_max_factor"].as<realtype>());
    bool error_outside_bounds = vm["error_outside_bounds"].as<bool>();
    std::vector<realtype> special_settings;
    auto special_settings_stream = std::istringstream(vm["special_settings"].as<std::string>());
    for (std::string item; std::getline(special_settings_stream, item, ',');){
        special_settings.push_back(std::atof(item.c_str()));
    }
    for (std::string line; std::getline(std::cin, line);){
        if (line.size() == 0)
            break;
        std::string item;
        auto linestream = std::istringstream(line);

        for (int idx=0; idx<ny; ++idx){
            std::getline(linestream, item, ' ');
            y0.push_back(std::atof(item.c_str()));
        }

        for (int idx=0; idx<nparams; ++idx){
            std::getline(linestream, item, ' ');
            params.push_back(std::atof(item.c_str()));
        }

        std::getline(linestream, item, ' ');
        dx0.push_back(std::atof(item.c_str()));

        std::getline(linestream, item, ' ');
        dx_min.push_back(std::atof(item.c_str()));

        std::getline(linestream, item, ' ');
        dx_max.push_back(std::atof(item.c_str()));

        for (; std::getline(linestream, item, ' ');)
            tout.push_back(std::atof(item.c_str()));

        if (nout == -1){
            nout = tout.size();
        }else if (static_cast<unsigned>(nout) != tout.size()){
            std::cerr << "Inconsistent length of tout" << std::endl;
            return 1;
        }

        systems.emplace_back(new odesys_anyode::OdeSys(&params[systems.size()*nparams], atol, rtol,
                                                       get_dx_max_factor, error_outside_bounds, special_settings));
    }
    if (nout < 2){
        std::cerr << "Got too few (" << nout << ") time points." << std::endl;
    } else if (nout == 2) {
        std::vector<realtype> t0;
        std::vector<realtype> tend;
        for (int idx=0; idx<systems.size(); ++idx){
            t0.push_back(tout[2*idx]);
            tend.push_back(tout[2*idx + 1]);
        }
        auto xy_ri = cvodes_anyode_parallel::multi_adaptive(
		systems, atol, rtol, lmm, &y0[0], &t0[0], &tend[0], mxsteps, &dx0[0],
		&dx_min[0], &dx_max[0], with_jacobian, iter_type, linear_solver, maxl,
		eps_lin, nderiv, return_on_root, autorestart, return_on_error);
    } else {
        std::vector<realtype> yout(systems.size()*ny);
        auto ri_ro = cvodes_anyode_parallel::multi_predefined(
		systems, atol, rtol, lmm, &y0[0], nout, &tout[0], &yout[0], mxsteps, &dx0[0],
		&dx_min[0], &dx_max[0], with_jacobian, iter_type, linear_solver, maxl,
		eps_lin, nderiv, autorestart, return_on_error);
    }
    for (auto& v : systems)
        delete v;
    return 0;
}
