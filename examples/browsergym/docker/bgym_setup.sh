# WA
WA_SESSION="webarena"
WA_SESSION_EXISTS=$(tmux list-sessions | grep $WA_SESSION)
if [ "$WA_SESSION_EXISTS" = "" ]
then
    tmux new-session -d -s $WA_SESSION

    tmux rename-window -t $WA_SESSION:0 "misc"

    for i in {1..5}
    do
        printf -v padded_i "%02d" "$i"
        tmux new-window -t $WA_SESSION:$i -n "wa_$padded_i"
        
        BASE_URL="http://localhost"
        tmux send-keys -t $WA_SESSION:$i "export WA_REDDIT='$BASE_URL:99$padded_i'" C-m
        
        tmux send-keys -t $WA_SESSION:$i "export WA_SHOPPING='$BASE_URL:7770'" C-m
        tmux send-keys -t $WA_SESSION:$i "export WA_SHOPPING_ADMIN='$BASE_URL:7780'" C-m
        tmux send-keys -t $WA_SESSION:$i "export WA_GITLAB='$BASE_URL:8023'" C-m
        tmux send-keys -t $WA_SESSION:$i "export WA_WIKIPEDIA='$BASE_URL:8898/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'" C-m
        tmux send-keys -t $WA_SESSION:$i "export WA_MAP='$BASE_URL:3000'" C-m
        tmux send-keys -t $WA_SESSION:$i "export WA_HOMEPAGE='$BASE_URL:4399'" C-m

        if (( i <= 5 )); then
            portion=$((i))
            tmux send-keys -t $WA_SESSION:$i "python llm-reasoners/examples/browsergym/inference_sglang.py --total-portions 5 --portion-idx $portion --mcts-iterations 10 --mcts-depth 5 --n-proposals 10" C-m
        elif (( i <= 10 )); then
            portion=$((i-5))
            tmux send-keys -t $WA_SESSION:$i "python llm-reasoners/examples/browsergym/inference_sglang.py --total-portions 5 --portion-idx $portion --mcts-iterations 10 --mcts-depth 10 --n-proposals 10" C-m
        else
            portion=$((i-10))
            tmux send-keys -t $WA_SESSION:$i "python llm-reasoners/examples/browsergym/inference_sglang.py --total-portions 10 --portion-idx $portion --mcts-iterations 10 --mcts-depth 20 --n-proposals 10" C-m
        fi
    done
fi
