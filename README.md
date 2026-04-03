# jaxgpt

The purpose of this repo is to showcase how you can build scalable LLMs in pure JAX, and train them in a **multi-host** environment.
The contained code showcases a GPT-2 like architecture, but includes modern architectural improvements and training methodologies:

* **Grouped Query Attention (GQA)**
* **No weight tying**
* **Rotary Position Embeddings (RoPE)**
* **QK-Norm**
* **Logits soft-capping**
* **ReLU-squared activations**
* **RMSNorm** (without learnable parameters)
* **Weight decay** (Optional)
* **Muon optimizer**
* **Cautious Weight Decay**


## Code Structure
The codebase is designed to be clean, modular, and easy to navigate without fighting heavy framework abstractions:

```text
jaxgpt/
├── gpt/                                         # Core GPT implementation in JAX
│   ├── __init__.py
│   ├── download_fineweb_tokens.py               # Utility to download tokenized Fineweb10B dataset
│   ├── fineweb_dataloader.py                    # Dataloader (uses `grain`)
|   ├── model.py                                 # Main GPT architecture definition
│   ├── layers.py                                # GPT core layers implementations
│   ├── optim.py                                 # Muon Optimizer
│   └── train.py                                 # Multi-host training loop and scaling logic
│   └── inference.py                             # Multi-host inference logic
│   └── kvcache.py                               # Rolling buffer KVCache
│   └── utils.py                                 # Minimal core abstraction for layers
│   └── config.py                                # Hyperparameters, sharding rules, and model specs
├── dev/
│   ├── instructions.md                          # Instructions to setup and run the code 
├── pyproject.toml                               # Project dependencies
└── README.md                                    # Repository documentation
```

<br>

## Getting Started

Please refer to [instructions.md](./dev/instructions.md) file for detailed instructions for running the code.

> [!IMPORTANT]
> If you modify the training or inference code, please keep in mind that you need to initialize the JAX distributed system by calling `jax.distributed.initialize()` (right after you import jax) on **all workers**, otherwise it won't work. 

<br>


## Results

After training the model for 5800 steps[^1], here are some samples of the generated text from the pretrained model.

```text
Prompt:
<|endoftext|>Did you notice that this world

Completion:
<|endoftext|>Did you notice that this world is the last chapter of the Warring States trilogy? Well, I thought so, and so, so did the author! They wrote the story for the book right down to the end. And boy, did they do it right. It went beautifully! I always wonder when the Dark Lord and the Chosen War would go into print. For me, the end is coming for a book that is as good as it gets.
First of all, I'm going to love all the stories in the
---------------------------------------------------------------------------

Prompt:
<|endoftext|>Hello World! My dear

Completion:
<|endoftext|>Hello World! My dear friend,
Well, I have written many of you asking questions about my situation, the first one being which should I wear a black jacket and shoes, and the second one being which should I wear shorts and shoes. Also, first question you ask is which outfit you wear your pants. Mine is black and the second question you ask, "If it looks like you, you should wear it."
I am trying to think about this question of "what kind of pants do you wear?" and
---------------------------------------------------------------------------

Prompt:
<|endoftext|>Some say we are tired far

Completion:
<|endoftext|>Some say we are tired far too fast.
And that is true.
I have been working in the music industry for over 14 years and have worked at all levels of the music industry.
I have a broad portfolio of music talent, including the likes of The Knife, James Brown and the likes of Johnny Cash, Robert Johnson, and Charlie Hunnam.
I have also worked in the entertainment industry, but have struggled with life changing events.
I have also worked in the music industry for over 21 years, and
---------------------------------------------------------------------------

Prompt:
<|endoftext|>What is AI?

Completion:
<|endoftext|>What is AI?
AI is the form of technology that provides information in order to interact with the world in a more effective manner. People are able to make decisions and act with their emotions, using computers and other forms of technology to change their behavior and have a better sense of security and safety. The internet has become the way people enjoy many different things, such as purchasing products, making shopping decisions, and having fun on social media. AI uses different types of technology to solve different problems, and it is a real
---------------------------------------------------------------------------
```
<br>

---

[^1]: *This code is built to be completely **accelerator agnostic**, meaning it can seamlessly run on both GPUs and TPUs. However, this was only tested on TPUv5p slices*. A massive thank you to the **Google Developer Experts (GDE) program** for generously providing the compute credits for this project.