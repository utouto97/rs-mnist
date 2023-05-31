#!/bin/bash

prefix="http://yann.lecun.com/exdb/mnist/"

function download() {
  (
    cd datasets
    curl -O "${prefix}$1"
    gzip -d $1
  )
}

download "train-images-idx3-ubyte.gz"
download "train-labels-idx1-ubyte.gz"
download "t10k-images-idx3-ubyte.gz"
download "t10k-labels-idx1-ubyte.gz"

