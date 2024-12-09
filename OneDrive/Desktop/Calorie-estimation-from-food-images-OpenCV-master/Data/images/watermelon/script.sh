#!/bin/bash
num=1
label="14_"
for file in *.JPG; do
       mv "$file" "$(printf "%s" $label$num).jpg"
       num=`expr $num + 1`
       echo $num
done

