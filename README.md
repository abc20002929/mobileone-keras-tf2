# mobileone-keras-tf2
mobileone tensorflow2/keras implemention
# feature
1.use DepthwiseConv2D instead of Conv2D(groups=N) directly.

2.tensorflow conv1x1+conv3x3 has different behavior with pytorch when stride=2.(the centers of 1x1 convolution and 3x3 convolution in TensorFlow are different)

3.remove deepcopy in reparameterize_modelTF, need copy outside.

# Usage
```python
from mobileone import mobileoneTF, reparameterize_modelTF

# To Train from scratch/fine-tuning
model = mobileone(variant='s0')
# ... train ...

# For inference  
model_eval = reparameterize_modelTF(model)
# Use model_eval at test-time
```

check correct reparameterize 
```python
from mobileone import mobileoneTF, reparameterize_modelTF

x = np.random.randn(1,224,224,3)

model = mobileone(variant='s0')
out1 = model(x)

model_eval = reparameterize_modelTF(model)
out2 = model_eval(xn)

print(np.sum((out1-out2).numpy()**2))

```

# Reference
https://github.com/apple/ml-mobileone/tree/main