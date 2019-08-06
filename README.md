# ShAIdes

My AI is so bright, I have to wear shades.

Coming soon!

### AWS

I trained on `g3s.xlarge` instances at Amazon AWS, with Nvidia Tesla M60 GPUs.  The `Deep Learning AMI (Ubuntu) Version 23.1` image has lots of machine learning packages preinstalled, so it's super easy to use.  Just launch an EC2 instance from the web dashboard, then clone my github repo:

```
git clone https://github.com/nickbild/shaides.git
```

Then start up a Python3 environment with PyTorch and dependencies and switch to my codebase:

```
source activate pytorch_p36
cd shaides
```

Now, run my training script (make sure to set the `classifier_type` and `class_count` variables to specify which model type you are training, and how many classes it contains):

```
python3 train.py
```

That's it!  Now, watch the output to see when the test accuracy gets to a good level (percentage in the 90s).

This will generate `*.model` output files from each epoch that you can download, e.g. with `scp` to use with the `infer_rt.py` script.

To train for your own purposes, just place your own images in the `data_objects` and `data_gestures` folders under `train` and `test`.  You can make your own folders there if you want to add new objects or gestures.
