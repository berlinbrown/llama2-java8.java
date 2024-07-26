## llama2.c

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file ([run.c](run.c)). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper). This repo is a "fullstack" train + inference solution for Llama 2 LLM, with focus on minimalism and simplicity.

As the architecture is identical, you can also load and inference Meta's Llama 2 models. However, the current code only inferences models in fp32, so you will most likely not be able to productively load models larger than 7B. Work on model quantization is currently ongoing.

Please note that this repo started recently as a fun weekend project: I took my earlier [nanoGPT](https://github.com/karpathy/nanoGPT), tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). So the project is young and moving quickly. Hat tip to the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. Compared to llama.cpp, I wanted something super simple, minimal, and educational so I chose to hard-code the Llama 2 architecture and just roll one inference file of pure C with no dependencies.

Original from: https://github.com/karpathy/llama2.c

Updated Jul 2024 Berlin Brown - with focus on training

## Feel the magic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karpathy/llama2.c/blob/master/run.ipynb)

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/karpathy/llama2.c.git
```

Then, open the repository folder:

```bash
cd llama2.c
```

Now, let's just run a baby Llama 2 model in C. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~60MB download):


You'll see the text stream a sample. On my M1 MacBook Air this runs at ~110 tokens/s. See [performance](#performance) or the Makefile for compile flags that can significantly speed this up. We can also try a bit bigger 42M parameter model:


This still runs at interactive rates and samples more coherent and diverse stories:

> Once upon a time, there was a little girl named Lily. She loved playing with her toys on top of her bed. One day, she decided to have a tea party with her stuffed animals. She poured some tea into a tiny teapot and put it on top of the teapot. Suddenly, her little brother Max came into the room and wanted to join the tea party too. Lily didn't want to share her tea and she told Max to go away. Max started to cry and Lily felt bad. She decided to yield her tea party to Max and they both shared the teapot. But then, something unexpected happened. The teapot started to shake and wiggle. Lily and Max were scared and didn't know what to do. Suddenly, the teapot started to fly towards the ceiling and landed on the top of the bed. Lily and Max were amazed and they hugged each other. They realized that sharing was much more fun than being selfish. From that day on, they always shared their tea parties and toys.

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

> One day, Lily met a Shoggoth. He was very shy, but was also very generous. Lily said “Hello Shoggy! Can I be your friend?” Shoggy was happy to have a friend and said “Yes, let’s explore the universe together!” So they set off on a journey to explore the universe. As they travelled, Shoggy was happy to explain to Lily about all the wonderful things in the universe. At the end of the day, Lily and Shoggy had gathered lots of wonderful things from the universe, and they both felt very proud. They promised to explore the universe as one big pair and to never stop being generous to each other.

There is also an even better 110M param model available, see [models](#models).

## Meta's Llama 2 models

As the neural net architecture is identical, we can also inference the Llama 2 models released by Meta. Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 7B model:

This ran at about 4 tokens/s compiled with [OpenMP](#OpenMP) on 96 threads on my CPU Linux box in the cloud. (On my MacBook Air M1, currently it's closer to 30 seconds per token if you just build with `make runfast`.) Example output:

> The purpose of this document is to highlight the state-of-the-art of CoO generation technologies, both recent developments and those in commercial use. The focus is on the technologies with the highest merit to become the dominating processes of the future and therefore to be technologies of interest to S&amp;T ... R&amp;D. As such, CoO generation technologies developed in Russia, Japan and Europe are described in some depth. The document starts with an introduction to cobalt oxides as complex products and a short view on cobalt as an essential material. The document continues with the discussion of the available CoO generation processes with respect to energy and capital consumption as well as to environmental damage.

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the chat model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

## Huggingface models

We can load any huggingface models that use the Llama 2 architecture. See the script [export.py](export.py) and the `--hf` flag to export the model .bin file.

## Training

When we say python, that is python3

Let's see how we can train a baby Llama 2 from scratch using the code in this repo. First let's download and pretokenize some source dataset, e.g. I like [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) so this is the only example currently available in this repo. But it should be very easy to add datasets, see the code.

This is an important step in getting the model built.

```bash
python tinystories.py download

python tinystories.py pretokenize
```

Let's see what happens at each step.

```bash
python tinystories.py download

```
When we run, it downloads the content for tiny stories:

```text
python3 tinystories.py download
Downloading https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz to data/TinyStories_all_data.tar.gz...
```

Code is really simply this:

```text
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
```

We can also replace text with our own custom data:

```bash
gsed -i 's/Lily/Berlin/g' data01.json 
gsed -i 's/Lily/Berlin/g' data01.json
```

The size of the zipped content is 1.5 gigs

Then we pretokenize it

```bash
python tinystories.py pretokenize
```

I am curious, just to run the model can we make it smaller so it can load in a could of hours or so.

Then train our model:

```bash
python train.py
```

After running the train, I got the followiing:

```text

    raise RuntimeError("Python 3.11+ not yet supported for torch.compile")
RuntimeError: Python 3.11+ not yet supported for torch.compile

With my mac, I guess torch compile was disabled
```

I ran the train and got the following with my small set:

```text
python3 -m train.py --compile=False --eval_iters=10 --batch_size=8

Overriding: compile = False
Overriding: eval_iters = 10
Overriding: batch_size = 8
tokens per iteration will be: 8,192
breaks down as: 4 grad accum steps * 1 processes * 8 batch size * 256 max seq len
Initializing a new model from scratch
num decayed parameter tensors: 43, with 15,187,968 parameters
num non-decayed parameter tensors: 13, with 3,744 parameters
using fused AdamW: False
Created a PretokDataset with rng seed 42
Created a PretokDataset with rng seed 42

...

And then
Between 9am to 12:45pm EST, the following

83 | loss 8.8303 | lr 4.150000e-05 | 24470.25ms | mfu 0.02%
84 | loss 8.8644 | lr 4.200000e-05 | 419462.74ms | mfu 0.01%
85 | loss 8.7363 | lr 4.250000e-05 | 7209674.88ms | mfu 0.01%
86 | loss 8.7896 | lr 4.300000e-05 | 48521.14ms | mfu 0.01%
87 | loss 8.7221 | lr 4.350000e-05 | 4755108.06ms | mfu 0.01%


88 | loss 8.7462 | lr 4.400000e-05 | 17563.82ms | mfu 0.01%
89 | loss 8.6417 | lr 4.450000e-05 | 14645.45ms | mfu 0.01%
90 | loss 8.6925 | lr 4.500000e-05 | 12714.74ms | mfu 0.01%
91 | loss 8.7091 | lr 4.550000e-05 | 18350.10ms | mfu 0.01%


Also there is a file in ./out/
ckpt.pt		model.bin

Testing example where my name is part of the data:

So we prime the data and then the response will include continuation based on the start:

input_text = "Lily and Ben built a fort"

Example output:

"Lily and Ben built a fort in the backyard. They used blankets, chairs, and cushions to create a cozy space. Inside, they imagined all sorts of adventures and spent the afternoon playing and telling stories."

Using the example input "Lily and Ben built a fort," the model generates a continuation of the story, adding context and details. This is useful for creating engaging content or extending prompts in creative ways.

Here I replaced Lily with Berlin

```text
 python3 sample.py --checkpoint=./out/ckpt.pt --start='Once upon a time there was little girl name'
Overriding: checkpoint = ./out/ckpt.pt
Overriding: start = Once upon a time there was little girl name
Once upon a time there was little girl name. She loved to play outside in the park with her dad your world, they asked her?"
As they played Tom were a beautiful, so she had lots of fun.
When they decided to see to the park, her dad was so excited candy that she was very sad, your soft ch named Berlin's mom, said, "Thank you, Berlin and tookak people to the end much fun by the bird." 
After she was very this and said, "
```

```

**brief training guide**. See the train.py script for more exotic launches and hyperparameter overrides. Here is a brief guide to how to set the parameters. Look at the table at the very end of the [Chinchilla paper](https://arxiv.org/abs/2203.15556) to get a sense of how the Transformer parameters (dim, n_layers, n_heads) grow or shrink together. Extrapolate/interpolate this pattern to get bigger or smaller transformers. Set the max context length however you wish, depending on the problem: this should be the max number of tokens that matter to predict the next token. E.g. Llama 2 uses 2048. Next, you want the _total_ batch size per update (printed by the script as "tokens per iteration will be:") to be somewhere around 100K tokens for medium-sized applications. For tiny applications it could be lower, for large training (e.g. GPTs/LLamas) it is usually ~0.5M, or even more. You get there by first maxing out the batch_size to whatever your system allows (e.g. mine was 16 in a recent run because after that my GPU runs out of memory), and then you want to increase gradient_accumulation_steps to be as high as necessary to reach the total batch size of ~100K. Finally, you want to tune your learning_rate (LR). You want this to be as high as your training allows. Very small networks can get away with a large LR (e.g. 1e-3 or even higher). Large networks need lower LRs. 3e-4 is a safe choice in most medium-sized applications, but can be too low for small networks, so try to increase it! Finally, max_iters is the length of training. Play with different settings. I mostly only ever tune these parameters and leave most of the others unchanged. Here is an example of how I trained the 110M model, which I don't think is anywhere near optimal, but looked sensible to me: dim 768, n_layers 12, n_heads 12 (so size of each head is 768 / 12 = 64 channels), seq len of 1024, batch size 16 (this is the most that fit my A100 40GB GPU), gradient_accumulation_steps = 8 was needed to get total tokens batch size to be 16 batch size * 1024 tokens in sequence * 8 grad_accum = 131,072 tokens per update. Good. Learning rate 4e-4 (probably a little too low). max_iters 200K (probably a bit too high). Dropout 0.1, as that usually helps a bit at medium size. That was it. I ran using Distributed Data Parallel (DDP) on 4 GPUs on my cloud machine, training took ~day or so.

Totally understand if you want to skip model training, for simple demo just download one of the pretrained models (see [models](#models) section), e.g.:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
make run
```

You can now run it simply as

```bash
./run stories15M.bin
```

Watch the tokens stream by, fun! We can also run the PyTorch inference script for a comparison. Download one of the models again from huggingface hub and point the `sample.py` script at it:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M

python sample.py --checkpoint=out15M/stories15M.pt
```

Which gives the same results.

## Custom tokenizers

In everything above, we've assumed the custom Lllama 2 tokenizer with 32,000 tokens. However, in many boutique LLMs, using vocabulary this big might be an overkill. If you have a small application you have in mind, you might be much better off training your own tokenizers. This can make everything nicer - with smaller vocabs your model has fewer parameters (because the token embedding table is a lot smaller), the inference is faster (because there are fewer tokens to predict), and your average sequence length per example could also get smaller (because the compression is a lot more efficient on your data). So let's see how we train a custom tokenizer.

By default, to pretokenize the tinystories dataset we had to run, in order:

```
python tinystories.py download

python tinystories.py pretokenize
```

The `pretokenize` stage here loads the Llama 2 tokenizer (vocab size 32,000) and uses it to convert the downloaded text into integers, and saves that to file. We now change this as follows, to train an example 4096-token tokenizer:

```
python tinystories.py download
python tinystories.py train_vocab --vocab_size=4096
python tinystories.py pretokenize --vocab_size=4096
```

The `train_vocab` stage will call the `sentencepiece` library to train the tokenizer, storing it in a new file `data/tok4096.model`. I tried to reproduce as well as I could the settings that (I think) Meta used to train their vocabulary. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary. Inspect the `tinystories.py` file - the custom tokenizers are stored in a special directory structure indexed by the vocab size.

A quick note of interest is that vocab size of 4096 trained specifically on tinystories creates integer sequences with about the same sequence length per example as the default Llama 2 tokenizer of 32000 tokens! This means that our custom, tailored tokenizer is a lot better adapted to our specific text, and can compress it very effectively. So our trained models are smaller and faster.

Now that we have pretokenized the dataset with our custom tokenizer, we can train the model. The training script `train.py` doesn't care about the exact tokens, it only cares about the vocabulary size so it can correctly initialize the model. So when training your model, make sure to pass in

```
python train.py --vocab_source=custom --vocab_size=4096
```

(The defaults are `llama2` and `32000` respectively, which indicates the default Llama 2 tokenizer). This trains the model. Finally we are ready to run inference with our `run.c` script. For that we need two things. Number one, we have to export our tokenizer in the `.bin` format, do that with:

```
python tokenizer.py --tokenizer-model=data/tok4096.model
```

This writes the tokenizer to `data/tok4096.bin`. Now we can run inference, pointing it to this tokenizer using the `-z` flag:

```
./run out/model.bin -z data/tok4096.bin
```

This should print the samples. If you leave out the `-z` flag, it will use the default Llama 2 tokenizer, which would generate a good sequence of integers, but they would get translated using a different vocabulary to text, so it would look like gibberish.



Depending on your system resources you may want to tweak these hyperparameters and use more threads. But more is not always better, usually this is a bit U shaped. In particular, if your CPU has SMT (multithreading), try setting the number of threads to the number of physical cores rather than logical cores. The performance difference can be large due to cache thrashing and communication overhead. The PyTorch documentation [CPU specific optimizations
](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations) has some good information that applies here too.

## platforms

On **Windows**, use `build_msvc.bat` in a Visual Studio Command Prompt to build with msvc, or you can use `make win64` to use mingw compiler toolchain from linux or windows to build the windows target. MSVC build will automatically use openmp and max threads appropriate for your CPU unless you set `OMP_NUM_THREADS` env.

On **Centos 7**, **Amazon Linux 2018** use `rungnu` Makefile target: `make rungnu` or `make runompgnu` to use openmp.

On **Mac**, use clang from brew for openmp build. Install clang as `brew install llvm` and use the installed clang binary to compile with openmp: `make runomp CC=/opt/homebrew/opt/llvm/bin/clang`.

## ack

I trained the llama2.c storyteller models on a 4X A100 40GB box graciously provided by the excellent [Lambda labs](https://lambdalabs.com/service/gpu-cloud), thank you.


## License

MIT
