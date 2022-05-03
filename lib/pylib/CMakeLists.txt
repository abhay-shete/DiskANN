# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_library(pydisk_index SHARED disk_index.cpp)
        target_link_libraries(pydisk_index ${PROJECT_NAME} -ltcmalloc)
 
 
