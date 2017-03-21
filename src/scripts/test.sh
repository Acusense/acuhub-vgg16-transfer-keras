#!/usr/bin/env bash

echo "testing long running process";

sleep 2

for i in {1..50}
do
   echo "hello world $i"
   sleep 2
done