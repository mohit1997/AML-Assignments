#!/bin/bash
s1="mnist"
s2="emnist"

if [ "$1" = "$s1" ]; then
  echo downloading mnist
  wget -A gz -R html,tmp  -r -l 1 -nd http://yann.lecun.com/exdb/mnist/
fi

if [ "$1" = "$s2" ]; then
  echo downloading emnist
  wget https://cloudstor.aarnet.edu.au/plus/index.php/s/54h3OuGJhFLwAlQ/download
fi
