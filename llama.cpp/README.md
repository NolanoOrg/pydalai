# llama.cpp

[![Actions Status](https://github.com/ggerganov/llama.cpp/workflows/CI/badge.svg)](https://github.com/ggerganov/llama.cpp/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Inference of [Facebook's LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++

## Description

The main goal is to run the model using 4-bit quantization on a MacBook

- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via Arm Neon and Accelerate framework
- AVX2 support for x86 architectures
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

This was [hacked in an evening](https://github.com/ggerganov/llama.cpp/issues/33#issuecomment-1465108022) - I have no idea if it works correctly.
Please do not make conclusions about the models based on the results from this implementation.
For all I know, it can be completely wrong. This project is for educational purposes.
New features will probably be added mostly through community contributions.

Supported platforms:

- [X] Mac OS
- [X] Linux
- [X] Windows (via CMake)

---

Here is a typical run using LLaMA-7B:

```java
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
I llama.cpp build info:
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

make: Nothing to be done for `default'.
main: seed = 1678486056
llama_model_load: loading model from './models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

main: prompt: 'Building a website can be done in 10 simple steps:'
main: number of tokens in prompt = 15
     1 -> ''
  8893 -> 'Build'
   292 -> 'ing'
   263 -> ' a'
  4700 -> ' website'
   508 -> ' can'
   367 -> ' be'
  2309 -> ' done'
   297 -> ' in'
 29871 -> ' '
 29896 -> '1'
 29900 -> '0'
  2560 -> ' simple'
  6576 -> ' steps'
 29901 -> ':'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000


Building a website can be done in 10 simple steps:
1) Select a domain name and web hosting plan
2) Complete a sitemap
3) List your products
4) Write product descriptions
5) Create a user account
6) Build the template
7) Start building the website
8) Advertise the website
9) Provide email support
10) Submit the website to search engines
A website is a collection of web pages that are formatted with HTML. HTML is the code that defines what the website looks like and how it behaves.
The HTML code is formatted into a template or a format. Once this is done, it is displayed on the user's browser.
The web pages are stored in a web server. The web server is also called a host. When the website is accessed, it is retrieved from the server and displayed on the user's computer.
A website is known as a website when it is hosted. This means that it is displayed on a host. The host is usually a web server.
A website can be displayed on different browsers. The browsers are basically the software that renders the website on the user's screen.
A website can also be viewed on different devices such as desktops, tablets and smartphones.
Hence, to have a website displayed on a browser, the website must be hosted.
A domain name is an address of a website. It is the name of the website.
The website is known as a website when it is hosted. This means that it is displayed on a host. The host is usually a web server.
A website can be displayed on different browsers. The browsers are basically the software that renders the website on the user’s screen.
A website can also be viewed on different devices such as desktops, tablets and smartphones. Hence, to have a website displayed on a browser, the website must be hosted.
A domain name is an address of a website. It is the name of the website.
A website is an address of a website. It is a collection of web pages that are formatted with HTML. HTML is the code that defines what the website looks like and how it behaves.
The HTML code is formatted into a template or a format. Once this is done, it is displayed on the user’s browser.
A website is known as a website when it is hosted

main: mem per token = 14434244 bytes
main:     load time =  1332.48 ms
main:   sample time =  1081.40 ms
main:  predict time = 31378.77 ms / 61.41 ms per token
main:    total time = 34036.74 ms
```

And here is another demo of running both LLaMA-7B and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) on a single M1 Pro MacBook:

https://user-images.githubusercontent.com/1991296/224442907-7693d4be-acaa-4e01-8b4f-add84093ffff.mp4

## Usage

Here are the step for the LLaMA-7B model:

```bash
# build this repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
python3 -m pip install torch numpy sentencepiece

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
./quantize.sh 7B

# run the inference
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 128
```

When running the larger models, make sure you have enough disk space to store all the intermediate files.

TODO: add model disk/mem requirements

### Interactive mode

If you want a more ChatGPT-like experience, you can run in interactive mode by passing `-i` as a parameter.
In this mode, you can always interrupt generation by pressing Ctrl+C and enter one or more lines of text which will be converted into tokens and appended to the current context. You can also specify a *reverse prompt* with the parameter `-r "reverse prompt string"`. This will result in user input being prompted whenever the exact tokens of the reverse prompt string are encountered in the generation. A typical use is to use a prompt which makes LLaMa emulate a chat between multiple users, say Alice and Bob, and pass `-r "Alice:"`.

Here is an example few-shot interaction, invoked with the command
```
./main -m ./models/13B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -i -r "User:" \
                                           -p \
"Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User:"

```
Note the use of `--color` to distinguish between user input and generated text.

![image](https://user-images.githubusercontent.com/1991296/224575029-2af3c7dc-5a65-4f64-a6bb-517a532aea38.png)

## Limitations

- I don't know yet how much the quantization affects the quality of the generated text
- Probably the token sampling can be improved
- The Accelerate framework is actually currently unused since I found that for tensor shapes typical for the Decoder,
  there is no benefit compared to the ARM_NEON intrinsics implementation. Of course, it's possible that I simlpy don't
  know how to utilize it properly. But in any case, you can even disable it with `LLAMA_NO_ACCELERATE=1 make` and the
  performance will be the same, since no BLAS calls are invoked by the current implementation

### Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo
- Collaborators will be invited based on contributions

### Coding guide-lines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy looking modern STL constructs, use basic for loops, avoid templates, keep it simple
- There are no strict rules for the code style, but try to follow the patterns in the code (indentation, spaces, etc.). Vertical alignment makes things more readable and easier to batch edit
- Clean-up any tailing whitespaces, use 4 spaces indentation, brackets on same line, `int * var`
- Look at the [good first issues](https://github.com/ggerganov/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks
