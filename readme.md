## Melsi

<img src="melsi%20screenshot.jpg" width="300">

Melsi is a sketch, audio, and text based educational web app. It uses an LLM to understand user questions and to generate answers. It replies with slides of short texts and sketches. A user can select the number of slides of answers they get. It supports PDF (<=5MB) uploads.

Melsi is designed to work for elementary school goers, but can be easily adapted for any age.

SVGs can be downloaded from the [Openart collection](https://drive.google.com/file/d/1_RFNcsB4u3WI2FyswIwvsV_Q08fxGnlb/view), and need to be stored in `./svg` directory.

The following setup worked on an NVIDIA GeForce RTX 4090. 

Start the model server:<br>
`vllm serve "./Llama-3-8B-instruct" --dtype float16 --gpu-memory-utilization 0.95 --max-model-len 8192 --max-num-seqs 32`

Start the backend:<br>
`uvicorn backend:app --host 0.0.0.0 --port 8502`

Then open in your browser:<br>
`http://localhost:8502`