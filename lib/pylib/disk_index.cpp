#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"
#include <iostream>

extern "C" {
    
    int build_index(const char* dataFilePath, const char* indexFilePath, const char* str_metric, const char* indexBuildParameters)
    {

      //const char* dataFilePath = "/Users/abhay.shete/Downloads/siftsmall/test_raw.bin";
      //const char* indexFilePath = "/Users/abhay.shete/code/DiskANN_data/bigann";
      //const char* indexBuildParameters = "32 50 .0005 .0010 32 0";
      //const char* indexBuildParameters = "32 50 .001 .07 32 0";

      diskann::Metric metric = diskann::Metric::L2;

      if (std::string(str_metric) == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;


      return diskann::build_disk_index<float>(dataFilePath, indexFilePath,
                                        indexBuildParameters, metric);
    }


    int search_index(const char* indexFilePath, const char* str_metric, const char* queryFilePath, const char* resFilePath, const char* indexBuildParameters, uint32_t num_neighbours)
    {
      diskann::Metric metric = diskann::Metric::L2;

      if (std::string(str_metric) == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;

      return diskann::search_disk_index<float>(indexFilePath, metric, 
                                    "float", queryFilePath, resFilePath, indexBuildParameters, num_neighbours);

    }
}