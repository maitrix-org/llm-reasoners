start_port="$1"
num_dockers="$2"
for i in $(seq 1 "$num_dockers")
do 
    ((current_port=start_port + i))
    docker stop forum_$current_port
    docker remove forum_$current_port
done
