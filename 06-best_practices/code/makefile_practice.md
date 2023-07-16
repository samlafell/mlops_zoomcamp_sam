
Without makefiles:
```bash
isort .
black .
pylint --recursive=y .
pytest tests/
```

How can we avoid having to do this manually?
- Git Hooks
    - Pre-commit hooks are run before commit
    - Python Tool: pre-commit (pip install)

    - Folder in .git that is called hooks
        - less pre-commit.sample (inside of hooks dir)
    
    - pre-commit library
        - helps us define the pre-commit hooks in Python

    
- Let's pretend that 06-bestpractices/code is its own folder
    - go to code folder, and run `git init`
    - pretend its a stand alone repo for the sake of the video

    