#!/bin/bash

while getopts n:e: option
do
	case ${option} in
		n) NET=${OPTARG};;
		e) EPS=${OPTARG};;
	esac
done

NET_NAME=$(cut -d '/' -f 3 <<< ${NET})
NET_NAME=$(cut -d '.' -f 1 <<< ${NET_NAME})
RESULTS_FILE=results_${NET_NAME}_eps_${EPS}.txt

rm ${RESULTS_FILE} 2>/dev/null || true

for i in {0..99}
do
	python3 analyzer.py ${NET} ../mnist_images/img${i}.txt ${EPS} >> ${RESULTS_FILE}
done
