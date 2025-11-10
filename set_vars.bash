#!/usr/bin/bash

if [ "$#" -ne 2 ]; then
	echo "./set_vars.bash <data path> <result path>"

else
	echo "setting vars"
	export _datapath=$1
	export _resultpath=$2
fi
