#!/bin/bash

prefix=passiv_en_nps
suffixes="train dev test gen"

for suffix in $suffixes; do
	python to_json.py $prefix.$suffix > $prefix.$suffix.json
done
