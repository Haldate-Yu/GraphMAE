device=$1

if [ -z "$1" ]; then
  echo "empty cuda input!"
  device=0
else
  device=$1
fi

export CUDA_VISIBLE_DEVICES=$device

for method in "zero" "random"; do
  python pretrain.py --predefine $method
done
