num_dockers="$1"
for i in $(seq 1 "$num_dockers")
do 
    printf -v padded_i "%02d" "$i"
    docker stop forum_$padded_i
    docker remove forum_$padded_i
    docker run --name forum_$padded_i -d -p 99$padded_i:80 postmill-populated-exposed-withimg
done
