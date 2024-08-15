cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름 및 버전
project(MainProgram VERSION 1.0)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TBB 라이브러리 찾기
find_package(TBB REQUIRED)

# 실행 파일 추가
add_executable(main main.cpp)

# TBB 라이브러리 링크
target_link_libraries(main TBB::tbb)
