# Master Experiment

All experiment files in one folder.

## Structure

```
master_experiment/
├── .env
├── .env.example
├── .gitignore
├── prompts/
│   ├── prompts.json
│   ├── few_shot_examples_grade_5.json
│   └── skills_description.json
├── config.py
├── run_experiment.py
├── README.md
└── experiments/          (output: experiment_001, experiment_002, ...)
```

## Setup

1. **API keys** - fill `master_experiment/.env`:
   ```
   DEEPSEEK_API_KEY=your_deepseek_key
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   ```

2. **Questions** - current search order:
   `../Data/UPDATED/`
   `../dataset/data99updated/`
   `../dataset/data99/`

3. **Current config**:
   grade: `5`
   model: `deepseek-chat`

## Run

From `master_experiment`:
```
python run_experiment.py
```

Or from Project folder:
```
python master_experiment/run_experiment.py
```

Results: `master_experiment/experiments/experiment_XXX/results_XXX.xlsx`
