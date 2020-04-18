# dotdot bot

AI Chatbot for use in dotdot. We're using the [transformers](https://github.com/huggingface/transformers) framework for the NLP

## Local Setup

### Create the Python virtual env

On the repo directory:

```console
python3 -m venv venv 
```

This should create the folder `venv` on the root of this repo, which should be ignored by git.
You can now move on to the next section

### Download models & config

Download the [DialoGPT 345M model, and the reverse](https://github.com/microsoft/DialoGPT#models) model and place in `medium/`
Download these [config files](https://github.com/microsoft/DialoGPT/tree/master/configs) and place in `config/`


## Local Development

Activate the virtual environment

```console
source venv/bin/activate
```

Install Dependencies (Only the first time you do this)

```console
pip install -r requirements.txt
```

Make sure pytorch is running well:

```console
python3 pytorch_check.py
```

And you should see on the screen:
```
tensor([[0.7917, 0.1953, 0.3718],
        [0.1171, 0.6210, 0.9340],
        [0.2153, 0.9759, 0.9654],
        [0.9698, 0.5383, 0.6687],
        [0.5058, 0.6857, 0.8104]])
Is CUDA Available?  False // OR True if available
```

Finally you can run:

```console
python3 interact.py
```