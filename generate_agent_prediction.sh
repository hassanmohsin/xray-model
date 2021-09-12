#!/usr/bin/env bash

for i in 1 2 3 4 5; do
  mkdir ./star/agents-performance/run-$i;
  python -m recommender.performance;
  mv ./star/agents-performance/agent* ./star/agents-performance/run-$i;
done
