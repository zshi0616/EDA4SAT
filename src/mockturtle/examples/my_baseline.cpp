#include <lorina/aiger.hpp>
#include <mockturtle/mockturtle.hpp>
#include <mockturtle/algorithms/lut_mapper.hpp>
#include <mockturtle/io/write_bench.hpp>
#include <../test/catch2/catch.hpp>

using namespace mockturtle;

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("[ERROR] Missing input or output filepath. \n");
    exit(0);
  }

  char *input_filepath = argv[1];
  char *output_filepath = argv[2];

  aig_network aig;
  auto const result = lorina::read_aiger(input_filepath, aiger_reader(aig) );

  if ( result != lorina::return_code::success ){
      std::cout << "Read benchmark failed\n";
      return -1;
  }

  mapping_view<aig_network, true> mapped_aig{aig};
  lut_map_params ps;
  ps.cut_enumeration_ps.cut_size = 4u;
  ps.cut_enumeration_ps.cut_limit = 8u;
  ps.recompute_cuts = true;
  ps.area_oriented_mapping = true;
  ps.cut_expansion = true;
  lut_map_stats st;

  lut_map<decltype( mapped_aig ), true>(mapped_aig, ps, &st);
  const auto klut = *collapse_mapped_network<klut_network>( mapped_aig );

  // Output
  depth_view<klut_network> klut_d{ klut };
  printf("Size %d, Depth: %d, Time: %.2f\n", klut.num_gates(), klut_d.depth(), to_seconds( st.time_total));
  write_bench(klut, output_filepath );

  return 0;
}