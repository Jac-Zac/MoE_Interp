- [Head persuite]

- MLP -> Molgeva
- Weight interpetrability of the weight (and also ROME)

## RL

- RL for LoRA -> Read LoRA without regrets
  https://www.youtube.com/watch?v=IdV5TEIsJhs

# NNsight

1. Phepras you can try to use vLLM infernece engine and test it with a small MoE model to actually see the basic performance
2. The next step would be to run inference on some prompts, eg 5 random prompts that you can import from a file or something. Or potentially laoding just one part of a dataset
3. Cache all activations and write them to disk every (tot time to avoid wasating things -> In this case I can write to the scratch partition)

Try also to only save one expert for example. Then read head persuit algorithm and try to do based on that

-> Be careful of using the correct template. If I need to add tokens or doing somehting with the prefix or something

# NOTE:

Review properties in python datacasses

- You can add alliasing like this https://github.com/ndif-team/nnsight/blob/4f2bdbe8aacb19b02a618ed222545b4d91e3b731/src/nnsight/intervention/envoy.py#L1339
