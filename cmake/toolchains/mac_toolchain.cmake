set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR arm64)

# Modify these variables with paths to appropriate compilers that can produce
# armv8 targets
set(CMAKE_C_COMPILER  /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_C_COMPILER_AR   
    /usr/bin/llvm-ar   
    CACHE FILEPATH "Archiver")
set(CMAKE_CXX_COMPILER_AR 
    /usr/bin/llvm-ar   
    CACHE FILEPATH "Archiver")

set(CMAKE_OSX_ARCHITECTURES arm64)
