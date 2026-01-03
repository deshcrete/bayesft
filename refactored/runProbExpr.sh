for i in {4}; do
    rm -rf ./data
    rm -rf ./lora_weights
    python3 ./refactored/main.py
    echo "Expr Num: $i"
done