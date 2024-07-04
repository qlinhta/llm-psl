#!/bin/bash

SESSION_NAME="qlta"

tmux start-server

tmux new-session -d -s $SESSION_NAME

tmux rename-window -t $SESSION_NAME:0 'Main'
tmux send-keys -t $SESSION_NAME:0 'cd ~' C-m

tmux new-window -t $SESSION_NAME:1 -n 'Editor'
tmux send-keys -t $SESSION_NAME:1 'cd ~ && vim' C-m

tmux new-window -t $SESSION_NAME:2 -n 'Logs'
tmux send-keys -t $SESSION_NAME:2 'cd /var/log' C-m

tmux split-window -h -t $SESSION_NAME:2
tmux send-keys -t $SESSION_NAME:2.1 'tail -f syslog' C-m

tmux new-window -t $SESSION_NAME:3 -n 'Server'
tmux send-keys -t $SESSION_NAME:3 'cd ~/my_server && ./start_server.sh' C-m

tmux select-window -t $SESSION_NAME:0

tmux attach-session -t $SESSION_NAME
