 #!/bin/bash

 nsys profile -o pk_10_log_repeats --stats=true -s none --trace=cuda,nvtx python run_pk_single_ufunc.py
