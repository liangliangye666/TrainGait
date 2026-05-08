#!/bin/bash

# rm -r install
rm -rf install build  # 一次性清理 install 和 build

cmake --fresh -B build -S .
time make install -j --no-print-directory -C build