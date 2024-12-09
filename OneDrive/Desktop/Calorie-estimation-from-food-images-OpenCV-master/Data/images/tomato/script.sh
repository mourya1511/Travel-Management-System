	#!/bin/bash
num=1
label="9_"
for file in *.jpg; do
       mv "$file" "$(printf "%s" $label$num).jpg"
       num=`expr $num + 1`
       echo $num
done

