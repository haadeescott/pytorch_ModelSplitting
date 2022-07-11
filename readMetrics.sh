#!/bin/bash
mpstat 1 100 > mpstatLog.txt &
free -s 1 -c 100 -k > freeLog.txt&
sudo powertop --time=10 --sample=10 --iteration=10 --csv=powerLogs/Log.txt&