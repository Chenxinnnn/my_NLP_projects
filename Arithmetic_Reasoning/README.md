# Prompt_Based_Addition

This project explores prompt engineering strategies for teaching large language models (LLaMA-2 via Together.ai API) to perform 7-digit integer addition. The work was conducted in the context of **Week 4** of the NLP course at NYU, and focuses on both theoretical and practical challenges in eliciting reliable arithmetic behavior from large pre-trained models using in-context learning.

## Folder Structure
```
Prompt_Based_Addition/
├── addition_prompting.ipynb        # Notebook for zero-shot and few-shot prompting
├── prompt.txt                      # Manually crafted prompts
├── submission.py                   # Final submission (API, prompt, config)
├── test_prompts.py                 # Evaluation script (local)
├── requirements.txt                # Required libraries
└── README.md                       # This file
```

## Task Description

The task is to accurately compute the sum of two random 7-digit integers using a language model, with no arithmetic logic hard-coded into the model. The training process is entirely prompt-based, using zero-shot and in-context learning techniques.

Example:

Input prompt:
Question: What is 1234567 + 2345678?
Answer:

Expected output:
3580245

This differs from standard supervised learning, as we only guide the model using prompt design without updating model parameters.


## Set up environment
We recommend you to set up a conda environment for packages used in this homework.
```
conda create -n 2590-hw4 python=3.9.7
conda activate 2590-hw4
pip install -r requirements.txt
```


## Approaches

The project is divided into the following stages:

- Zero-shot prompting
- In-context learning (ICL)
- Parameter tuning: max_tokens, temperature, top_p, top_k, stop
- Leaderboard submission with accuracy and MAE scores

## Results

| Prompt Type        | Max Tokens | Accuracy | MAE              |
|--------------------|------------|----------|------------------|
| Zero-shot (70B)    | 8          | ~0.00    | >7,000,000       |
| 1-shot (7-digit)   | 8          | ~0.33    | ~2,000,000       |
| 1-shot (7-digit)   | 50         | 0.40     | <1,000,000       |

Best performing prompt uses a 7-digit in-distribution example and sets max_tokens = 50.

## References

- Scratchpads for Intermediate Computation (Nye et al., 2021)
- Prompt Engineering Guide: https://www.promptingguide.ai/
- Together.ai Playground: https://api.together.xyz/playground

## Tools & Environment

- Python 3.10+
- Together.ai API
- Jupyter Notebook
- LLaMA-2 7B and 70B

## Key Takeaways

- Prompt design directly impacts arithmetic accuracy.
- In-distribution examples drastically improve few-shot performance.
- Output token length (max_tokens) has a major effect on performance bias.


