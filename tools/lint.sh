#!/bin/zsh
# Unused-code check for the shader. Clang reports unused statics and constants only in the main file, and Saver.metal
# includes everything, so this inlines the includes into one file (build/lint.metal) and checks that.
here=${0:A:h}
root=${here:h}
cd $root/Shaders || exit 1
mkdir -p $root/build
inline() {
  local line f n=0
  while IFS= read -r line; do
    (( n++ ))
    if [[ $line =~ '^#include "([^"]+)"' ]]; then
      f=${match[1]}
      print -r -- "# 1 \"Shaders/$f\""
      cat $f; print
      print -r -- "# $((n + 1)) \"Shaders/$1\""
    else
      print -r -- $line
    fi
  done < $1
}
inline Saver.metal > $root/build/lint.metal
cd $root
xcrun metal -fsyntax-only -std=metal3.1 -Wunused -Wunused-const-variable build/lint.metal
