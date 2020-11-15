#include <cstdlib>
#include <sysexits.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp>
#include <cvodes_anyode_parallel.hpp>

namespace po = boost::program_options;

${p_odesys_impl}

typedef ${p_realtype} realtype;
typedef ${p_indextype} indextype;

int main(int argc, char *argv[]){
    // Parse cmdline args:
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("atol", po::value<realtype>()->default_value(1e-8), "absolute tolerance")
        ("rtol", po::value<realtype>()->default_value(1e-8), "relative tolerance")
        ("mxsteps", po::value<int>()->default_value(500), "maximum number of steps")
        ("lmm", po::value<std::string>()->default_value("BDF"), "linear multistep method")
        ("return-on-root", po::value<bool>()->default_value(false), "Return on root")
        ("autorestart", po::value<int>()->default_value(0), "Autorestart (autonomous)")
        ("return-on-error", po::value<bool>()->default_value(false), "Return on error")
        ("with-jtimes", po::value<int>()->default_value(0), "With jtimes")
        ("get-dx-max-factor", po::value<realtype>()->default_value(1.0), "get_dx_max multiplicative factor")
        ("error-outside-bounds", po::value<bool>()->default_value(false), "Return recoverable error to solver when outside bounds")
        ("max-invariant-violation", po::value<realtype>()->default_value(0.0), "Limit at which to return recoverable error when supported by integrator.")
        ("special-settings", po::value<std::string>()->default_value(""), "special settings2 (if customized)")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    try {
        po::notify(vm);
    } catch (boost::exception& exc) {
        std::cerr << boost::diagnostic_information(exc);
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return EX_USAGE;
    }

    const indextype ny = ${p_odesys.ny};
    const int nparams = ${len(p_odesys.params)};
    std::vector<odesys_anyode::OdeSys<realtype, indextype> *> systems;
    const std::vector<realtype> atol(ny, vm["atol"].as<realtype>());
    const realtype rtol(vm["rtol"].as<realtype>());
    const auto lmm((vm["lmm"].as<std::string>() == "adams") ?
                   cvodes_cxx::LMM::Adams :
                   cvodes_cxx::LMM::BDF);

    std::vector<realtype> y0;

    std::vector<realtype> tout;
    int nout = -1;
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
    bool return_on_root(vm["return-on-root"].as<bool>());
    int autorestart(vm["autorestart"].as<int>());
    bool return_on_error(vm["return-on-error"].as<bool>());
    bool with_jtimes(vm["with-jtimes"].as<int>());

    std::vector<realtype> params;
    const realtype get_dx_max_factor(vm["get-dx-max-factor"].as<realtype>());
    bool error_outside_bounds = vm["error-outside-bounds"].as<bool>();
    const realtype max_invariant_violation(vm["max-invariant-violation"].as<realtype>());
    std::vector<realtype> special_settings;
    auto special_settings_stream = std::istringstream(vm["special-settings"].as<std::string>());
    for (std::string item; std::getline(special_settings_stream, item, ',');){
        special_settings.push_back(std::atof(item.c_str()));
    }
    // Parse stdin:
    int rowi=0;
    for (std::string line; std::getline(std::cin, line); ++rowi){
    // nt y0[0] ... y0[ny - 1] params[0] ... params[nparams - 1] t[0] ... t[nout - 1]
        if (line.size() == 0)
            break;
        std::string item;
        auto linestream = std::istringstream(line);

        std::getline(linestream, item, ' ');
        const int nt = std::atoi(item.c_str());

        for (indextype idx=0; idx<ny; ++idx){
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
        }
        if (tout.size() != (rowi+1)*nt or nt != nout){
            std::cerr << "Inconsistent length of tout" << std::endl;
            return 1;
        }

        systems.push_back(new odesys_anyode::OdeSys<realtype, indextype>(&params[systems.size()*nparams],
                                                                         atol, rtol, get_dx_max_factor,
                                                                         error_outside_bounds,
                                                                         max_invariant_violation,
                                                                         special_settings));
    }
    // Computations:
    int nprealloc = 500;
    int * td_arr;
    realtype ** xyout_arr;
    std::vector<std::pair<int, std::vector<int> > > n_ri;
    std::vector<std::pair<int, std::pair<std::vector<int>, std::vector<realtype> > > > ri_ro;
    std::vector<realtype> yout;

    if (nout < 2){
        std::cerr << "Got too few (" << nout << ") time points." << std::endl;
    } else if (nout == 2) {
        xyout_arr = (realtype **)malloc(systems.size()*sizeof(realtype*));
        for (int i=0; i<systems.size(); ++i) {
            xyout_arr[i] = (realtype*)malloc((ny+1)*nprealloc*sizeof(realtype));
            td_arr[i] = nprealloc;
        }
        std::vector<realtype> t0;
        std::vector<realtype> tend;
        for (int idx=0; idx<systems.size(); ++idx){
            tend.push_back(tout[2*idx + 1]);
            xyout_arr[idx][0] = tout[2*idx];
            for (indextype iy=0; iy<ny; ++iy){
                xyout_arr[idx][iy + 1] = y0[ny*idx + iy];
            }
        }
        n_ri = cvodes_anyode_parallel::multi_adaptive(
	    xyout_arr, td_arr, systems, atol, rtol, lmm, &tend[0], mxsteps, &dx0[0],
            &dx_min[0], &dx_max[0], with_jacobian, iter_type, linear_solver, maxl,
            eps_lin, nderiv, return_on_root, autorestart, return_on_error, with_jtimes
        );


    } else {
        yout.resize(systems.size()*ny*nout);
        ri_ro = cvodes_anyode_parallel::multi_predefined(
        systems, atol, rtol, lmm, &y0[0], nout, &tout[0], &yout[0], mxsteps, &dx0[0],
        &dx_min[0], &dx_max[0], with_jacobian, iter_type, linear_solver, maxl,
        eps_lin, nderiv, autorestart, return_on_error, with_jtimes);
    }
    // Output:
    for (indextype si=0; si<systems.size(); ++si){
        bool first = true;
        for (int pi=0; pi<nparams; ++pi){
            if (first)
                first = false;
            else
                std::cout << ' ';
            std::cout << params[si*nparams + pi];
        }
        std::cout << '\n';
        if (nout == 2) {
            for (int ti=0; ti <= n_ri[si].first; ++ti){
                std::cout << xyout_arr[si][(1+ny)*ti];
                for (indextype yi=0; yi<ny; ++yi)
                    std::cout << ' ' << xyout_arr[si][(1+ny)*ti+yi+1];
                std::cout << '\n';
            }
        } else {
            for (indextype ti=0; ti<nout; ++ti){
                std::cout << tout[si*nout + ti];
                for (indextype yi=0; yi<ny; ++yi){
                    std::cout << ' ' << yout[si*nout*ny + ti*ny + yi];
                }
                std::cout << '\n';
            }
        }
        std::cout << '{';
        first = true;
        for (const auto& itm : systems[si]->current_info->nfo_int) {
            if (first){
                first = false;
            } else {
                std::cout << ", ";
            }
            std::cout << "'" << itm.first << "': " << itm.second;
        }
        for (const auto& itm : systems[si]->current_info.nfo_dbl) {
            std::cout << ", '" << itm.first << "': " << itm.second;
        }
        for (const auto& itm : systems[si]->current_info.nfo_vecdbl) {
            std::cout << ", '" << itm.first << "': {";
            first = true;
            for (const auto& elem : itm.second) {
                if (first){
                    first = false;
                } else {
                    std::cout << ", ";
                }
                std::cout << elem;
            }
            std::cout << "}";
        }
        for (const auto& itm : systems[si]->current_info.nfo_vecint) {
            std::cout << ", '" << itm.first << "': {";
            first = true;
            for (const auto& elem : itm.second) {
                if (first){
                    first = false;
                } else {
                    std::cout << ", ";
                }
                std::cout << elem;
            }
            std::cout << "}";
        }

        std::cout << "}\n";
    }

    for (auto& v : systems)
        delete v;
    return EX_OK;
}
