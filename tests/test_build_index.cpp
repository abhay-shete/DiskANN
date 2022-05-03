#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"


extern "C" {
    
    int build_index(const char* dataFilePath, const char* indexFilePath)
    {

      //const char* dataFilePath = "/Users/abhay.shete/Downloads/siftsmall/test_raw.bin";
      //const char* indexFilePath = "/Users/abhay.shete/code/DiskANN_data/bigann";
      const char* indexBuildParameters = "32 50 .0005 .0010 32 0";
      diskann::Metric metric = diskann::Metric::L2;

      return diskann::build_disk_index<float>(dataFilePath, indexFilePath,
                                      indexBuildParameters, metric);
    }


    int search_index() {
      const std::string queryFilePath = "/Users/abhay.shete/Downloads/siftsmall/test_raw.bin";
      const std::string indexFilePath = "/Users/abhay.shete/code/DiskANN_data/bigann/2/bigann";

      diskann::Metric metric = diskann::Metric::L2;

      return diskann::search_disk_index<float>(indexFilePath, metric, 
                                    "float", queryFilePath, "/tmp/res_lib");

    }

}