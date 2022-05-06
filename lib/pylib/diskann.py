import numpy as np
import math
import string
import os
import tempfile
import ctypes
import struct
from pathlib import Path


class IndexBuildParams:
    def __init__(self, metric="inner_product", graph_degree=100, search_list_size=150,
                 max_mem_search=0.3,
                 max_mem_build=0.5,
                 num_threads=32,
                 store_compressed=False):
        """
        Parameters
        ----------
        metric: str
            One of "inner_product" or "l2"

        graph_degree: int
            The degree of the graph index, typically between 60 and 150.
            Larger R will result in larger indices and longer indexing times,
            but better search quality.

        search_list_size: int
            the size of search list during index build. Typical values are between 75 to 200.
            Larger values will take more time to build but result in indices that provide
            higher recall for the same search complexity. Use a value for L value that is at
            least the value of R unless you need to build indices really quickly and
            can somewhat compromise on quality.

        max_mem_search: float
            bound on the memory footprint of the index at search time in GB. Once built, the index
            will use up only the specified RAM limit, the rest will reside on disk.
            This will dictate how aggressively we compress the data vectors to store in memory.
            Larger will yield better performance at search time. For an n point index, to use b byte
            PQ compressed representation in memory,
            use B = ((n * b) / 2^30 + (250000*(4*R + sizeof(T)*ndim)) / 2^30).
            The second term in the summation is to allow some buffer for caching about 250,000 nodes
            from the graph in memory while serving.
            If you are not sure about this term, add 0.25GB to the first term
        max_mem_build: float
            Limit on the memory allowed for building the index in GB.
            If you specify a value less than what is required to build the index in one pass,
            the index is built using a divide and conquer approach so that sub-graphs will fit in the RAM budget.
            The sub-graphs are overlayed to build the overall index. This approach can be upto 1.5 times slower
            than building the index in one shot. Allocate as much memory as your RAM allows.

        num_threads: int
            Number of threads used by the index build process. Since the code is highly parallel,
            the indexing time improves almost linearly with the number of threads
            (subject to the cores available on the machine and DRAM bandwidth).

        store_compressed: bool
            Use False to store uncompressed data on SSD. This allows the index to asymptote to 100% recall.
            If your vectors are too large to store in SSD, this parameter provides the option to compress
            the vectors using PQ for storing on SSD. This will trade off recall.
            You would also want this to be greater than the number of bytes used for the PQ compressed data
            stored in-memory

        """
        self.metric = "mips"
        if metric == "l2":
            self.metric = "l2"
        self.R = graph_degree
        self.L = search_list_size
        self.B = max_mem_search
        self.M = max_mem_build
        self.T = num_threads
        self.pq_disk_type = store_compressed

    def __str__(self):
        """
        This function is overridden so that it returns the build parameters
        as required by the diskann api. Hence it does not include metric
        """
        return "{} {} {} {} {}".format(self.R, self.L, self.B, self.M, self.T, self.pq_disk_type)


class IndexSearchParams:
    def __init__(self, num_nodes_to_cache=200000, num_threads=10, beam_width=4,
                 search_list_size=20):
        """
        Parameters
        ----------

        num_nodes_to_cache: int
            While serving the index, the entire graph is stored on SSD.
            For faster search performance, you can cache a few frequently
            accessed nodes in memory. Higher value provides better recall.

        num_threads: int
            Number of parallel search threads

        beam_width: int
            The  beam width to be used for search. This is the maximum number of
            IO requests each query will issue per iteration of search code.
            Larger beam width will result in fewer IO round-trips per query,
            but might result in slightly higher total number of IO requests to SSD
            per query. For the highest query throughput with a fixed SSD IOps rating,
            use beam_width=1. For best latency, use beam_width=4,8 or higher complexity search

        search_list_size: int
            Size of the node search list during search. Higher value results in better recall
            at the cost of query throughput. Range varied from k+10 to k+40 for the SIFT 1 million
            dataset, where k is the number of near neighbours

        """
        self.num_nodes_to_cache = num_nodes_to_cache
        self.num_threads = num_threads
        self.beam_width = beam_width
        self.L = search_list_size

    def __str__(self):
        return "{} {} {} {}".format(self.num_nodes_to_cache,
                                    self.num_threads,
                                    self.beam_width,
                                    self.L)


class DiskANN:
    def __init__(self, lib_path, index_path=None, metric="mips"):
        self.lib_path = lib_path
        self.handle = ctypes.CDLL(lib_path)
        self.handle.build_index.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.handle.search_index.argtypes = [ctypes.c_char_p,
                                             ctypes.c_char_p,
                                             ctypes.c_char_p,
                                             ctypes.c_char_p,
                                             ctypes.c_char_p,
                                             ctypes.c_uint]
        self.sizeof_uint = 4
        if index_path is not None:
            self.index_path = str.encode(index_path)
        self.metric = metric

    def save_bin_file(self, vecs):
        header = np.array([vecs.shape[0], vecs.shape[1]])
        f = tempfile.NamedTemporaryFile(delete=False)
        header.astype("int32").tofile(f)
        for v in vecs:
            v.astype("float32").tofile(f)
        f.close()
        return f.name

    def load_query_results(self, bin_file_path, num_near_neighbours, fmt_type="I"):
        res = []
        with open(bin_file_path, "rb") as f:
            num_entries = struct.unpack('i', f.read(4))[0]
            dims = struct.unpack('i', f.read(4))[0]
            for i in range(num_entries):
                B = f.read(num_near_neighbours * self.sizeof_uint)
                res += struct.unpack(str(num_near_neighbours) + fmt_type, B)
        os.remove(bin_file_path)
        return res

    def build_disk_index(self, index_path, vecs, index_params=None):
        parent_dir = str(Path(index_path).parent)
        os.makedirs(parent_dir, exist_ok=True)
        vec_bin_path = str.encode(self.save_bin_file(vecs))
        self.index_path = str.encode(index_path)
        if index_params is None:
            index_params = IndexBuildParams()
        self.metric = index_params.metric
        index_params_str = str.encode(str(index_params))
        return self.handle.build_index(vec_bin_path, self.index_path,
                                       str.encode(index_params.metric), index_params_str)

    def search_disk_index(self, query, n, index_search_params=None):
        """
        Params:
        ======
        query: numpy array
            query vectors

        n: int
            Number of near neighbours

        index_search_parameters: IndexSearchParameter object
            Class object containing default search parameters.

        Returns:
        =======
        array
            Array of tuples containing id and distance
        """
        query_file_path = self.save_bin_file(query)
        res_prefix = "/tmp/query_res"
        if index_search_params is None:
            index_search_params = IndexSearchParams()

        self.handle.search_index(self.index_path,
                                 str.encode(self.metric),
                                 str.encode(query_file_path),
                                 str.encode(res_prefix),
                                 str.encode(str(index_search_params)),
                                 n)
        ids_path = res_prefix + "_{}_idx_uint32.bin".format(index_search_params.L)
        dists_path = res_prefix + "_{}_dists_float.bin".format(index_search_params.L)

        res_ids = self.load_query_results(ids_path, n)
        dists = self.load_query_results(dists_path, n, "f")
        os.remove(query_file_path)
        return [(res_id, dist) for res_id, dist in zip(res_ids, dists)]
