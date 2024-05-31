#!/bin/bash

prefix=counterfactual_question_have_de
suffixes="train dev test gen"

for suffix in $suffixes; do
	python to_json.py $prefix.$suffix > $prefix.$suffix.json
done
