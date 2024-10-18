# LLaVA-NDiNO

 🤗📚 [Datasets](https://huggingface.co/collections/swap-uniba/lvlm-italian-data-67099957a6ad85d9c8fbccb7) | 🤗💻 [Models](https://huggingface.co/collections/swap-uniba/llava-ndino-670906c04aba6241d52df43d)

Repository for the paper "LLaVA-NDiNO: Empowering LLMs with Multimodality for the Italian Language"

## Introduction

LLaVA-NDiNO is a family of models trained for optimized performance in the Italian language. Specifically, the models have been trained using three different approaches (either only one of them or by applying them in sequence):

- Language Adaptation: by pre-training the model on a rich collection of image-text data
- Instruction-Tuning: by fine-tuning the model on instruction-following image-text data (where the model answer is brief)
- Long Instruction-Tuning: by fine-tuning the model on instruction-following image-text data (where the model answer is long)

In this repository we provide everything we used for training and evaluation.
Please note that this work used the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) codebase for the training procedure. We modified a single script, we provide this script in the repository.

## Repository Structure

- 📁 [lmms-eval-tasks](lmms-eval-tasks): contains the tasks implementations to be added to the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) library to reproduce the evaluation results on the Italian versions of GQA, POPE, SeedBENCH, OK-VQA, MTVQA and EXAMS-V
- 📁 [requirements](requirements): contains the Singularity definition file to build the Singularity container used for the training step
- 📄 [convert_llava_weights.py](convert_llava_weights.py): script used to convert the LLaVA-NeXT checkpoint obtained by the original codebase into the HuggingFace format
- 📄 [evaluate.sh](evaluate.sh): template script to evaluate the models on the Italian versions of GQA, POPE, SeedBENCH, OK-VQA, MTVQA and EXAMS-V
- 📄 [evaluate_ppl.py](evaluate_ppl.py): script to evaluate the models on the Perplexity metric
- 📄 [llava_train_modified.py](llava_train_modified.py): modified [train script of the original LLaVA-NeXT repository](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/train/train.py) to apply the LLaMA 3 chat template without system prompt
- 📄 [train_from_llm.sh](train_from_llm.sh): template script to train a LLaVA-NeXT model from a pre-trained LLM
- 📄 [train_from_lmm.sh](train_from_lmm.sh): template script to train a LLaVA-NeXT model from a pre-trained LLaVA-NeXT model

## Usage

To train a model, you should:
- Build the Singularity container using the definition file in [requirements](requirements)
- Replace the original [train.py](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/train/train.py) script with the [llava_train_modified.py](llava_train_modified.py) script
- Perform the LLaVA-NDiNO train steps,[train_from_llm.sh](train_from_llm.sh) and [train_from_lmm.sh](train_from_lmm.sh) are template scripts to train LLaVA-NeXT starting from a LLM and a LLaVA-NeXT checkpoint respectively
- Convert the model using the [convert_llava_weights.py](convert_llava_weights.py) script

To evaluate a model, you should:
- Clone and install the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) library
- Add the task folders and the mBlip script in [lmms-eval-tasks](lmms-eval-tasks) to the [tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks) and [models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) directories respectively
- Evaluate the models following the template scripts in [evaluate.sh](evaluate.sh)

## Citation

```
@inproceedings{musacchioLLaVANDiNO,
  title={LLaVA-NDiNO: Empowering LLMs with Multimodality for the Italian Language},
  author={Musacchio, Elio and Siciliani, Lucia and Basile, Pierpaolo and Semeraro, Giovanni},
  booktitle={Proceedings of the Eighth Workshop on Natural Language for Artificial Intelligence (NL4AI 2024) co-located with 23th International Conference of the Italian Association for Artificial Intelligence (AI*IA 2024)},
  year={2024}
}
```
