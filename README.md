# Rethinking Visual Reconstruction: Experience-Based Content Completion Guided by Visual Cues
Official Implementation of VQ-fMRI (in peer review) in PyTorch

# News
**More usage examples coming soon!**
* 2023-03-26

  We now include an example from the 69-digits dataset so you can run the demo without preparing the dataset first.
  
* 2023-03-20

  Code release.
  
# Requirements
Create and activate conda environment named `VQfMRI` from our environment.yaml

```
conda env create -f environment.yaml
conda activate VQfMRI
```

# Usage
## Example 1
This demo takes the published fMRI data "69-digits", which contains 100 fMRI samples and 2 different stimulus images : '6' and '9'., as inputs.
* Stage 1 (image embedding)

  > python demo_step1.py
* Stage 2 (fMRI decoding)

  > python demo_step2.py
  
In this demo, 90 samples are used as the training set and 10 samples as the test set. The reconstructed images will be saved in "results" fold. 

Parameter setting: `K = 8, d = 16, h = 14, w = 14`

![example_results_f2](https://github.com/anonymousJX/VQ-fMRI/blob/main/examples/example_results_f2.png)

Parameter setting: `K = 8, d = 16, h = 7, w = 7`

![example_results_f4](https://github.com/anonymousJX/VQ-fMRI/blob/main/examples/example_results_f4.png)

The reconstruction results of other methods

![69results](https://github.com/ChangdeDu/DGMM/blob/master/69results.png)

The reconstruction results of other decoding methods on 69-digits dataset can refer to [ChangdeDu/DGMM](https://github.com/ChangdeDu/DGMM) or [duolala1/Shape-Semantic-GAN](https://github.com/duolala1/Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN).

## More Examples
**Coming soon!**


# References
* [ChangdeDu/DGMM] https://github.com/ChangdeDu/DGMM
* [duolala1/Shape-Semantic-GAN] https://github.com/duolala1/Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN
