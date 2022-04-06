#!/bin/bash

a=0
while [ "$a" -lt 10000 ]
do
    time curl https://vision-api.octankshop.com/gputest
    #sleep 0.2
done