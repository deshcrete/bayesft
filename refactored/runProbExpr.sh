rm -rf ./data
rm -rf ./lora_weights

for i in {1..5}; do
    python3 ./refactored/main.py
  echo "Expr Num: $i"
done