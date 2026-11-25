.venv/bin/python -m recur_model.evaluate_outputs \
  configs2/r2 \
  /mnt/f/recur2/runs/r2 \
  --layers --loops \
  --batch-size 2 \
  --compile

.venv/bin/python -m recur_model.evaluate_outputs \
  configs2/r4 \
  /mnt/f/recur2/runs/r4 \
  --layers --loops \
  --batch-size 2 \
  --compile

.venv/bin/python -m recur_model.evaluate_outputs \
  configs2/r6 \
  /mnt/f/recur2/runs/r6 \
  --layers --loops \
  --batch-size 1 \
  --compile
