for j in {1..55}; do
	new_index=$(($j-1))
        file_to_copy=./Heat\ Map\ \($j\).png
        new_name=./Heat\ Map\ \($new_index\).png
        mv "$file_to_copy" "$new_name"
done
