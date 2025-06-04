# Overview

LightinGAN is a collection of a generation model using Pix2PixHD, and an object detection model YOLOv11 by Ultralytics. The generator model takes an input as the image file and the value of lighting scenario (1-24), and outputs the image with the applied lighting scenario as the output, of size 256x256.

# Install

:warning: Please ensure you have suitable hardware and the supported OS to run the code. 

Supported OS: Windows, Linux, macOS.
Packages required to be installed: They are clearly defined in the requirements.txt file.
Hardware supported: Any graphics card that supports CUDA. RTX 20 series and above are recommended for inferences and training with fewer epochs.

**Command to install the recommended packages**
```bash
pip install -r requirements.txt
```


# Usage Example
## Generation model for generating images

Input: Image file along with the lighting scenario you want to create it for.
Output: Image file with the required lighting scenario applied for that image.


*For more information about the dataset, see:
Multi-illumination dataset from CSAIL of MIT [https://projects.csail.mit.edu/illumination/]


# Citing LightinGAN (for Generative model or Detection model)

If you use LightinGAN, please cite the following work:

```LaTeX
@inproceedings{lightingan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

---
