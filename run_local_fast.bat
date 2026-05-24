@echo off
setlocal
python scripts\check_gpu.py
python scripts\local_run_all.py --suite all --profile fast --batch_size 8 --num_workers 0
endlocal
