#include <argparse/argparse.hpp>
#include <cstdio>
#include <string>

#include "spark/random/random.h"
#include "simulation.h"
#include "simulation_events.h"

spark::Parameters get_case_parameters(int case_number) {
    spark::Parameters p;

    switch (case_number) {
        case 1:
            p = spark::Parameters::case_1();
            break;
        case 2:
            p = spark::Parameters::case_2();
            break;
        case 3:
            p = spark::Parameters::case_3();
            break;
        case 4:
            p = spark::Parameters::case_4();
            break;
        default:
            break;
    }

    return p;
}

int main(int argc, char* argv[]) {
    spark::random::initialize(500);
    argparse::ArgumentParser args("spark-benchmark");

    int case_number = 0;
    args.add_argument("case_number")
        .help("Benchmark case to be simulated")
        .scan<'i', int>()
        .default_value(1)
        .choices(1, 2, 3, 4)
        .store_into(case_number);

    std::string data_path{"../data"};
    args.add_argument("-d", "--data")
        .help("Path to folder with cross section data")
        .default_value(data_path)
        .store_into(data_path);

    args.parse_args(argc, argv);

    printf("Starting benchmark case %d simulation\n", case_number);
    printf("Data path set to %s\n", data_path.c_str());

    spark::Simulation sim(get_case_parameters(case_number), data_path);
    spark::setup_events(sim);
    sim.run();

    return 0;
}