cmake_minimum_required(VERSION 3.18)

# 프로젝트 이름 및 CUDA 설정
project(CUDA_TBB_Example LANGUAGES CXX CUDA)

# CUDA 컴파일러 설정
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 라이브러리 찾기
find_package(CUDA REQUIRED)

# TBB 라이브러리 찾기
find_package(TBB REQUIRED)

# 소스 파일 설정
set(SOURCES main.cu)

# 실행 파일 생성
add_executable(main ${SOURCES})

# TBB 및 CUDA 라이브러리 링크
target_link_libraries(main TBB::tbb ${CUDA_LIBRARIES})

# CUDA include directories 설정
include_directories(${CUDA_INCLUDE_DIRS})
