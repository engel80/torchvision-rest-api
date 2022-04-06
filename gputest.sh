#!/bin/bash

a=0
while [ "$a" -lt 100 ]
do
    time curl http://127.0.0.1:9000/gputest
    sleep 1
done