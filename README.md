# Physical Reasoning in Vision-Language Models

This repository contains the code for my undergraduate senior thesis at the University of Illinois Urbana-Champaign:

**“Physical Reasoning in Vision-Language Models”** :

## Overview

This project investigates whether improved performance of vision-language models (VLMs) on physical reasoning benchmarks reflects genuine physical understanding.

Using hardness-aware finetuning on Physion and custom Physion-style datasets, the work evaluates models including:
- InternVL2-8B
- Qwen2.5-VL-7B-Instruct

The repository includes:
- Physion and custom dataset evaluation pipelines
- Hardness-aware finetuning code
- Multi-view reasoning experiments
- Color and texture robustness analysis
- Geometry, time, and contact probing experiments
- Result analysis and visualization scripts

## Main Findings

The experiments show that:
- Finetuning improves benchmark accuracy on Physion-style tasks
- Improvements do not transfer well to multi-view 3D reasoning
- Models are not strongly relying on simple color or texture shortcuts
- Auxiliary training on geometry, time, and contact does not improve physical prediction performance

Overall, the results suggest that higher benchmark accuracy does not necessarily imply genuine physical reasoning.

## Thesis

The full thesis document will soon be included in this repository.

## Author

Ieva Bagdonaviciute  
University of Illinois Urbana-Champaign  
Advisor: David Forsyth
