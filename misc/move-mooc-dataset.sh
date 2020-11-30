#!/bin/sh

for i in {0..55}; do
    for j in {0..55}; do
        path=pwd
        new_directory="$(pwd)/data3/$j"
        old_directory="$(pwd)/data/hm$i"
        mkdir -p $new_directory
        file_to_copy=$old_directory/Heat\ Map\ \($j\).png
        destination=$new_directory/$i.png
        echo "$file_to_copy"
        echo "$destination"
        cp "$file_to_copy" "$destination"
    done
done
