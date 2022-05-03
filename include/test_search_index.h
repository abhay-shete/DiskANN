// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
//#include "test_search_index.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false


template<typename T>
inline int search_disk_index(std::string index_prefix_path, std::string str_metric, std::string data_type, std::string query_bin, 
                        std::string result_output_prefix, uint64_t num_nodes_to_cache=1000, uint32_t num_threads=10, uint32_t beamwidth=4, uint32_t num_results=20) {
  // load query bin
  T*                query = nullptr;
  size_t            query_num, query_dim, query_aligned_dim;

  diskann::Metric metric = diskann::Metric::L2;

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }


  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  //_u64        recall_at = std::atoi(argv[ctr++]);


  // TODO: Convert this to a parameter...
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
#endif

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader, metric));

  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str());

  if (res != 0) {
    return res;
  }
  // cache bfs levels
  std::vector<uint32_t> node_list;
  diskann::cout << "Caching " << num_nodes_to_cache
                << " BFS nodes around medoid(s)" << std::endl;
  //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  _pFlashIndex->generate_cache_list_from_sample_queries(
       warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
  _pFlashIndex->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_ids(1);
  std::vector<std::vector<float>>    query_result_dists(1);


  _u64 L = num_results;
  int recall_at = num_results;
  int test_id = 0;

  query_result_ids[test_id].resize(recall_at * query_num);
  query_result_dists[test_id].resize(recall_at * query_num);

  diskann::QueryStats* stats = new diskann::QueryStats[query_num];

  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
  auto                  s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (_s64 i = 0; i < (int64_t) query_num; i++) {
    _pFlashIndex->cached_beam_search(
        query + (i * query_aligned_dim), recall_at, L,
        query_result_ids_64.data() + (i * recall_at),
        query_result_dists[test_id].data() + (i * recall_at),
        beamwidth, stats + i);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  float qps = (1.0 * query_num) / (1.0 * diff.count());

  diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                             query_result_ids[test_id].data(),
                                             query_num, recall_at);

  diskann::cout << "Done searching. Now saving results " << std::endl;
  std::string cur_result_path =
      result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
  diskann::save_bin<_u32>(cur_result_path, query_result_ids[0].data(),
                          query_num, recall_at);

  cur_result_path =
      result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
  diskann::save_bin<float>(cur_result_path,
                           query_result_dists[0].data(), query_num,
                           recall_at);
  
  diskann::aligned_free(query);
  return 0;
}


int seearch_index_float(std::string index_prefix_path, std::string str_metric, std::string data_type, std::string query_bin, 
                        std::string result_output_prefix) {
    return search_disk_index<float>(index_prefix_path, str_metric, data_type, query_bin, result_output_prefix);
}