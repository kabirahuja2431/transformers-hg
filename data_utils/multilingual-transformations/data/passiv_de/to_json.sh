#!/bin/bash

prefix=passiv_de
suffixes="train dev test gen_rc_o gen_rc_s"

for suffix in $suffixes; do
	python to_json.py $prefix.$suffix > $prefix.$suffix.json
done
