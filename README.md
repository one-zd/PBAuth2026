# [Towards rigorous protection against unauthorized image synthesis on diffusion models]


Official implementation of **[Towards rigorous protection against unauthorized image synthesis on diffusion models]**.

> **[Towards rigorous protection against unauthorized image synthesis on diffusion models]**<br>
> Zhangdong Wang, Tongqing Zhou, Tengfei Zheng, Jiaohua Qin and Zhiping Cai <br>
>
---

</div>

![Method](assets/method_diagram.png) *(Placeholder for Method Diagram)*

---

<br>

</div>

## Contents
  - [Setup](#setup)
  - [Usage](#usage)
    - [Adversarial Encoding](#adversarial-encoding-protection)
    - [Web UI](#web-ui)
    - [Benchmarking](#benchmarking)
  - [Project Structure](#project-structure)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

<br>

## Setup

We recommend using Conda to manage the environment.

```bash
# 1. Create the environment from the provided yaml file
conda env create -f environment.yaml

# 2. Activate the environment
conda activate PBAuth_env
```

## Usage

### Adversarial Encoding (Protection)

The core script `PBAuth.py` is used to generate the protected (adversarial) images.

**Basic Usage:**

```bash
python PBAuth.py --input_path "path/to/benign_image.png" --output_path "path/to/protected_image.png"
```

**Advanced Usage with Robustness:**

```bash
python PBAuth.py \
  --input_path "example/input/1.png" \
  --output_path "example/output/1_protected.png" \
  --epsilon 0.1255 \
  --steps 200 \
  --color_loss_weight 1
```

### Web UI

PBAuth includes a Gradio-based Web UI for interactive testing.

```bash
python PBAuth.py --web_ui
```

### Benchmarking

Use `PBAuth_wbench.py` to evaluate the method on the W-Bench dataset or similar benchmarks.

```bash
python PBAuth_wbench.py --input_dir "path/to/dataset" --output_dir "path/to/results"
```

## Project Structure

- `PBAuth.py`: Main script for generating adversarial examples.
- `private_safety_checker.py`: Custom differentiable safety checker implementation.
- `training_src/`: Core training modules and transformation layers for robust optimization.
- `vine_safety/`: additional utilities for safety checks.
- `api/`: Scripts for interacting with Replicate API (SDXL/Inpainting) for testing.

## Acknowledgement

We benchmark our method using **[W-Bench](https://github.com/Shilin-LU/VINE#w-bench)**, a comprehensive benchmark for evaluating watermarking robustness against image editing, and utilize **[Replicate](https://replicate.com/)** for API-based testing.

If you use the datasets or baselines from these works, please refer to their respective repositories:
- **W-Bench Dataset details**: [HuggingFace W-Bench](https://huggingface.co/datasets/Shilin-LU/W-Bench)
- **VINE Official Repo**: [GitHub](https://github.com/Shilin-LU/VINE)

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{wang2026PBAuth,
  title={Towards rigorous protection against unauthorized image synthesis on diffusion models},
  author={Zhangdong Wang, Tongqing Zhou, Tengfei Zheng, Jiaohua Qin and Zhiping Cai},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2026},
  doi={10.1109/TCSVT.2026.3694225}
}
```
