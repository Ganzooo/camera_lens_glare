# Static shape
./bin/trtexec --explicitBatch \
          --onnx=./bin/glare_dua_mish_ema33_02.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bin/glare_dua_mish_ema33_02.engine