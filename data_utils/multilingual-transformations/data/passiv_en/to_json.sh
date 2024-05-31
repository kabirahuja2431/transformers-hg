#!/bin/bash

prefix=passivization
suffixes="train dev test gen"

for suffix in $suffixes; do
	python to_json.py $prefix.$suffix > $prefix.$suffix.json
done
