# NLP_project
Repo for Natural Language Processing project

## Usage
1. Install required packages
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Run evaluation script
```
python3 src/eval.py
```
3. Run the gradio UI
```
python3 src/ui.py
```

### Development
1. Install development packages
```
pip install -r requirements-dev.txt`
```
2. Run **mypy** (static code checker), **flake8** (linter) and **pytest** (unit tests)
```
sh check_code_quality.sh
```


