#!/bin/sh
python3 server/main.py &
jupyter-lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password="argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$X1hRpZbS3gQ09WAawc/rwg\$NZoj3vphgpAUrkwNC7c2OLeiKmspbBgdKulmbiVr2UE" --notebook-dir='/data' --NotebookApp.allow_origin='*'