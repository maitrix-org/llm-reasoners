num_dockers=$1

for i in $(seq 1 "$num_dockers")
do 
    printf -v padded_i "%02d" "$i"
    docker stop forum_$padded_i
    docker remove forum_$padded_i
done
