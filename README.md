# Java 8 port and verbose version of llama2.java

Make this very wordy and verbose
Also implement with Java8 for pure research purposes.

Also see llama2c-verbose - fork of: 

Fun with llm



## Notes on training

This is a fork and taken from the llama2.c docs, where they say "I", that is really from: https://github.com/karpathy/llama2.c

See this article on
https://arxiv.org/abs/2305.07759

https://arxiv.org/pdf/2305.07759

Also see paper

https://arxiv.org/abs/2203.15556

"Training Compute-Optimal Large Language Models" -- Deepmind, 2022

Download 

The file is about 1.5 gigs

https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz

Before running the python scripts, ensure to install against the requirements file.

pip3 install -r requirements.txt 

python3 tinystories.py download


```text
Unpacking data/TinyStories_all_data.tar.gz...
Download done.
Number of shards: 50
Example story:
{'story': '\n\nLily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n"Push me, Ben!" she says. Ben pushes her gently. Lily feels happy.

```

After download there are files in directory: TinyStories_all_data

```python

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")
```

See process_shard

Totally understand if you want to skip model training, for simple demo just download one of the pretrained models.

"The pretokenize stage here loads the Llama 2 tokenizer (vocab size 32,000) and uses it to convert the downloaded text into integers, and saves that to file. We now change this as follows, to train an example 4096-token tokenizer:"

"The train_vocab stage will call the sentencepiece library to train the tokenizer, storing it in a new file data/tok4096.model. I tried to reproduce as well as I could the settings that (I think) Meta used to train their vocabulary. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary."

python tinystories.py download
python tinystories.py train_vocab --vocab_size=4096
python tinystories.py pretokenize --vocab_size=4096   (This command takes a while)

python train.py --vocab_source=custom --vocab_size=4096

I ran train with this:

python3 -m train.py --compile=False --eval_iters=10 --batch_size=8

python tokenizer.py --tokenizer-model=data/tok4096.model

This writes the tokenizer to data/tok4096.bin. Now we can run inference, pointing it to this tokenizer using the -z flag:

./run out/model.bin -z data/tok4096.bin

If we look at the /llama2c-verbose/data/TinyStories_all_data data files

/llama2c-verbose/data/TinyStories_all_data/data00.json

Look like a list of stories sentences

[{"story": "\n\nLily and Ben are friends. They like to play in the park. One day, they see ...

Here is running with train:

```text
Berlins-MacBook-Pro:llama2c-verbose berlinbrown$ python3 -m train.py --compile=False --eval_iters=10 --batch_size=8
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
Created a PretokDataset with rng seed 42
step 0: train loss 10.4150, val loss 10.4116

This step takes many hours

```

### Python Libraries in Training

https://github.com/google/sentencepiece/blob/master/python/README.md

"Python wrapper for SentencePiece. This API will offer the encoding, decoding and training of Sentencepiece."

https://github.com/google/sentencepiece

"SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing."

```python
class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""
```

### More on python scripts

```python
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")
```

### Additonal Links

Also see: https://huggingface.co/blog/how-to-train


## A Java port of Andrej Karpathy's llama2.c (original document)

****Check the successor of this project: [Llama3.java](https://github.com/mukel/llama3.java): Practical Llama (3) inference in a single Java file, with additional features, including a `--chat` mode.**

This is a pure Java port of Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c), a very simple implementation
to run inference of models with a [Llama2](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.  

<p align="center">
  <img width="600" src="https://github.com/mukel/llama2.java/assets/1896283/66a8a650-f1a9-4540-9587-b112294e5e6b">
</p>

Currently, there isn't anything really original here, but I'll continue polishing it while keeping it in sync with the original.  
Besides the educational value, this project will be used to test and tune compiler optimizations on the JVM, particularly for the [Graal compiler](https://www.graalvm.org/latest/reference-manual/java/compiler).
This port used [llama2.scala](https://github.com/jrudolph/llama2.scala) initially as a reference.

## Build
Java 21+ is required, in particular the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).  

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) in the current directory.
You can use [TinyStories](https://huggingface.co/karpathy/tinyllamas/tree/main) checkpoints or get LLama2 models by [following instructions](https://github.com/karpathy/llama2.c#metas-llama-2-models).

```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

To build and run manually:
```bash
javac --enable-preview -source 21 --add-modules=jdk.incubator.vector Llama2.java
java --enable-preview --add-modules=jdk.incubator.vector Llama2 stories15M.bin
```

Or run it directly with [JBang](https://www.jbang.dev/):
```bash
jbang Llama2.java stories15M.bin
# With additional -D options and custom Java home.
JAVA_HOME=/path/to/java/home jbang -Djava.util.concurrent.ForkJoinPool.common.parallelism=0 -Dllama2.VectorAPI=false Llama2.java stories15M.bin
```

A `Makefile` and a `run.sh` script are also provided:

```bash
make # optional, run.sh already runs make

JAVA_HOME=$GRAALVM_HOME \
JAVA_RUNTIME_OPTIONS=-Djava.util.concurrent.ForkJoinPool.common.parallelism=8 \
./run.sh stories15M.bin
```

#### Native image

A standalone native image can be created with [GraalVM](https://www.graalvm.org/)
```bash
JAVA_HOME=$GRAALVM_HOME NATIVE_IMAGE_OPTIONS="-march=native" make native-image
./llama2 stories15M.bin
```

Or can also be built with [Profile-Guided Optimizations (PGO)](https://www.graalvm.org/dev/reference-manual/native-image/guides/optimize-native-executable-with-pgo), on Oracle GaaalVM:
```bash
JAVA_HOME=$GRAALVM_HOME \
NATIVE_IMAGE_OPTIONS="--pgo-instrument -march=native --initialize-at-build-time=Llama2 -Dllama2.VectorAPI=false" \
make native-image

# Profile run to generate default.iprof, with no parallelism to speedup profiling.
./llama2 -Djava.util.concurrent.ForkJoinPool.common.parallelism=0 stories15M.bin

# Build optimized image
JAVA_HOME=$GRAALVM_HOME \
NATIVE_IMAGE_OPTIONS="--pgo -march=native --initialize-at-build-time=Llama2 -Dllama2.VectorAPI=false" \
make native-image

# Should run ~2X faster than regular image.
./llama2 stories15M.bin
```

## Performance

Quick numbers on an AMD Ryzen 3950X 64GB, Arch Linux.  
`llama2.java` executed on OpenJDK 20.0.2+9.  
To make things fair w.r.t. to vectorization, the Java version has a matmul implementation using the [Vector API](https://openjdk.org/jeps/448).  
In these measurements the JVM is warmed up enough to reach peak tokens/s.  
On GraalVM, please note that the Graal compiler doesn't support the Vector API yet, to avoid unexpected performance degradation, run with `-Dllama2.VectorAPI=false`.

****Notes**  
*The numbers below were collected using aggressive (gcc) compiler flags e.g. regular `gcc -O2 ...` wouldn't be as fast.*

### Single-threaded

`llama2.c` compiled with `gcc -Ofast -march=native run.c -lm -o run -march=native`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=0`

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- | 
| stories15M.bin  |   363 |  1.0 | llama2.c    |
| stories15M.bin  |   237 | 0.65 | llama2.java |
| stories110M.bin | 51.71 |  1.0 | llama2.c    |
| stories110M.bin | 42.20 | 0.81 | llama2.java |
| llama2_7B.bin   |  0.92 |  1.0 | llama2.c    |
| llama2_7B.bin   |  0.88 | 0.95 | llama2.java |

### Multi-threaded

`llama2.c` compiled with `gcc -Ofast -fopenmp -march=native run.c -lm -o run -march=native`  
`llama2.c` executed with `OMP_NUM_THREADS=8`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=8`  

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- |
|  stories15M.bin |  1233 |  1.0 | llama2.c    |
|  stories15M.bin |   438 | 0.35 | llama2.java |
| stories110M.bin |    90 |  1.0 | llama2.c    |
| stories110M.bin |    80 | 0.88 | llama2.java |
|   llama2_7B.bin |  1.68 |  1.0 | llama2.c    |
|   llama2_7B.bin |  1.65 | 0.98 | llama2.java |

****Notes**  
*In `stories15M.bin`, the C version shows a huge speedup, very likely a cache effect, this is considered an outlier.
Running with 16/32 threads may actually cause a slowdown; the performance is, in most cases, U-shaped w.r.t to the # of threads.
With that many threads, vectorization does not give any advantage, since throughput is limited by memory bandwidth.*

Performance is already comparable to the original C code, bar vectorization, even if the Java code has not been optimized yet.

## License

MIT

### Additional Notes, Example json file

```json

[
   {
      "story":"\n\nLily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n\"Push me, Ben!\" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n\"Can I swing too, Lily?\" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n\"Ow, ow, ow!\" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben's hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.",
      "instruction":{
         "prompt:":"Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb \"hang\", the noun \"foot\" and the adjective \"cute\". The story has the following features: the story should contain at least one dialogue. Remember to only use simple words!\n\nPossible story:",
         "words":[
            "hang",
            "foot",
            "cute"
         ],
         "features":[
            "Dialogue"
         ]
      },
      "summary":"Lily and Ben play in the park and Lily gets too caught up in swinging, causing Ben to leave. Lily falls off the swing and hurts herself, but Ben leaves his hat for her as a kind gesture.",
      "source":"GPT-4"
   },
   {
      "story":"Once upon a time, there was a little girl named Lily. She had a teddy bear that she loved so much. One day, she lost it while playing in the park. She looked everywhere, but she couldn't find it. She felt sad and scared without her teddy bear. \nLily's mommy saw her crying and asked what was wrong. Lily told her that she lost her teddy bear. Mommy hugged her and said, \"Don't worry, we'll search for it together.\" They went back to the park and looked everywhere. After a while, they found the teddy bear under a tree. Lily was so happy! \nShe hugged her teddy bear and felt comfortable again. She said, \"I hope I never lose you again, teddy bear.\" Mommy smiled and said, \"Me too, Lily. You and teddy bear are the best of friends.\" And they all went home, happy and content. The end.",
      "instruction":{
         "prompt:":"Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. In the story, try to at some point use the verb \"hope\", the noun \"search\" and the adjective \"comfortable\". Remember to only use simple words!",
         "words":[
            "hope",
            "search",
            "comfortable"
         ],
         "features":[
            
         ]
      },
      "summary":"Lily loses her teddy bear in the park and her mommy helps her find it, making Lily happy again.",
      "source":"GPT-3.5"
   },

   ```