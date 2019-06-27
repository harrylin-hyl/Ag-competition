pstree $1 -p| awk -F"[()]" '{print $2}' | xargs kill -9
