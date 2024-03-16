#include <lorina/aiger.hpp>
#include <mockturtle/mockturtle.hpp>
#include <mockturtle/algorithms/lut_mapper.hpp>
#include <mockturtle/io/write_bench.hpp>
#include <../test/catch2/catch.hpp>
#include <../examples/parameter.hpp>

using namespace mockturtle;

float get_tt_score(kitty::dynamic_truth_table const& tt)
{
  float score = 0.0;
  if (tt._num_vars == 4) {
    score = SCORE_LIST_FANIN4[tt._bits[0]];
  }
  else if (tt._num_vars == 3) {
    score = SCORE_LIST_FANIN3[tt._bits[0]];
  }
  else if (tt._num_vars == 2) {
    score = SCORE_LIST_FANIN2[tt._bits[0]];
  }
  else {
    score = 0;
  }
  return 128.0 - score;
}

struct lut_custom_cost
{
  std::pair<uint32_t, uint32_t> operator()( kitty::dynamic_truth_table const& tt ) const
  {
    // if ( tt.num_vars() < 2u )
    //   return { 0u, 0u };
    // return { tt.num_vars(), 1u }; /* area, delay */
    float score = get_tt_score(tt);
    return {score, score}; /* area, delay */
  }
};

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

  // TODO: true or false
  // The template LUTCostFn sets the cost function to evaluate depth and size of a truth table given its support size, if StoreFunction is set to false, or its function, if StoreFunction is set to true.
  lut_map<decltype( mapped_aig ), true, lut_custom_cost>(mapped_aig, ps, &st);

  const auto klut = *collapse_mapped_network<klut_network>( mapped_aig );

  // Output
  depth_view<klut_network> klut_d{ klut };
  printf("Size %d, Depth: %d, Time: %.2f\n", klut.num_gates(), klut_d.depth(), to_seconds( st.time_total));
  write_bench(klut, output_filepath );

  return 0;
}