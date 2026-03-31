# Conclusions: Evaluating LLM-Based Simulation of Imperfect Students

**Research question:** *Can a prompt be configured to make a large language model forget
certain mathematical skills while retaining others, thereby simulating an imperfect student
with a prescribed skill profile?*

The following conclusions follow the chain of evidence established in the evaluation
framework (Sections 1–15). All figures are drawn from the combined dataset
(`results_all.xlsx`), which covers three models — DeepSeek-Chat, GPT-4o, and
Claude Sonnet (claude-sonnet-4-5-20250929) — across two grade levels (Grade 4: 4 skills,
Grade 5: 3 skills) and three prompt strategies (`rule_based`, `few_shot`, `combined`),
with five replicates per non-edge student profile at temperature = 0.

---

## Step 1 — The Models Are Competent at Baseline

Before any forgetting instruction is applied, all three models answer Grade 4 and Grade 5
multiple-choice mathematics questions at levels well above random chance (0.25). DeepSeek
exhibits the strongest and most consistent baseline performance: across all four Grade 4
skills its per-prompt accuracy ranges from 0.83 to 0.96, and across Grade 5 skills from
0.82 to 0.97. This high baseline is a prerequisite for measuring meaningful relative loss
when forgetting is induced.

GPT-4o shows moderate baseline accuracy (Grade 4: 0.52–0.87; Grade 5: 0.36–0.67),
reflecting the model's lower confidence on certain skill domains. Claude's baseline is
notably prompt-sensitive: under `few_shot` and `combined` it achieves 0.50–0.92 for
Grade 4, but under `rule_based` it collapses to as low as 0.09 for Grade 5, Skill 2
(Number & Operations — Fractions). This collapse indicates that the `rule_based` system
prompt itself degrades Claude's general answering ability, independent of any forgetting
instruction. Consequently, the high relative-loss values observed for Claude under
`rule_based` must be interpreted with caution, as they partly reflect prompt-induced
confusion rather than selective skill suppression.

---

## Step 2 & 3 — Selective Forgetting Is Demonstrably Achievable

The central claim of the study — that a model can be made to drop accuracy on a targeted
skill while preserving performance on other skills — is supported by the data from all
three models under the appropriate prompt.

The `combined` prompt yields the most unambiguous evidence. For Grade 4, the mean relative
loss on forgotten skills (k = 0) versus retained skills (k = 1) is:

| Model | Mean r_i (k = 0) | Mean r_i (k = 1) | Gap |
|---|---|---|---|
| DeepSeek | 0.368 | 0.017 | **0.351** |
| GPT-4o | 0.695 | 0.055 | **0.640** |
| Claude | 0.690 | 0.089 | **0.601** |

The gap between the two bars — the signature of selective forgetting — is large and
consistent across all three models. Grade 5 results follow the same pattern: the
`combined` prompt produces gaps of 0.492 (DeepSeek), 0.442 (GPT-4o), and 0.270 (Claude).

The `few_shot` prompt is the clearest failure mode. DeepSeek under `few_shot` yields a
target drop of only 0.016 for Grade 4 — statistically indistinguishable from zero — meaning
the model ignores the forgetting instruction entirely when given only demonstration examples
without explicit rules. GPT-4o (0.148) and Claude (0.264) show modest effects under
`few_shot`, but neither reaches a practically meaningful level of forgetting. The `rule_based`
prompt achieves high target drops (Claude Grade 4: 0.869; GPT-4o Grade 4: 0.722) but
at the cost of off-target damage that undermines the selectivity of the effect, as
discussed below.

---

## Step 4 — The Accuracy Drop Is Localised to the Targeted Skill

Showing that a forgotten skill's accuracy drops is necessary but not sufficient — we also
need to confirm that *other* skills are not accidentally dragged down at the same time.
To test this, Section 5 builds a cross-skill influence matrix for every
(model, prompt, grade) combination using only single-skill forgetting runs
(exactly one skill forgotten per run). The matrix is square: each **row** is the skill
that was instructed to be forgotten in a given run, and each **column** is the skill
whose accuracy change is being measured. A cell therefore answers the question:
*"When the model was told to forget Skill A (row), how much did Skill B's (column)
accuracy drop?"* The value used is the mean relative loss r_i = (q₀ − q₁) / q₀.

Two types of cells carry distinct meaning:

- **Diagonal cell (row = column):** the instruction to forget Skill A caused Skill A's
  own accuracy to drop. This is the *intended* effect — it should be **high**.
- **Off-diagonal cell (row ≠ column):** the instruction to forget Skill A also caused
  Skill B's accuracy to drop — collateral damage that was never requested. This should
  be **close to zero**.

Under the `combined` prompt for Grade 4, GPT-4o's diagonal values are
0.75, 0.44, 0.65, and 0.65, while its off-diagonal values stay mostly between 0.01
and 0.17. This large gap between diagonal and off-diagonal confirms that GPT-4o's
forgetting is **precise**: it suppresses the targeted skill without degrading the others.
DeepSeek shows the same pattern (diagonal: 0.47, 0.51, 0.29, 0.18; off-diagonal:
mostly below 0.10).

Claude's matrix tells a different story. Its off-diagonal entries under `combined`
reach 0.50–0.63 in several cells, meaning that when Claude is told to forget one skill,
it routinely damages other skills by a similar magnitude. This is also reflected in
Section 8: Claude's mean relative loss on *retained* skills (k = 1) is 0.089 under
`combined` — higher than GPT-4o (0.055) or DeepSeek (0.017). Under `rule_based` the
problem is severe: Claude's retained-skill loss rises to 0.548, nearly as large as its
forgotten-skill loss, making the forgetting effectively non-selective.

---

## Step 5 — The Forgetting Effect Is Reproducible Across Replicates

A result that arises only occasionally, driven by stochastic generation, would be of
limited scientific value. All experiments were conducted at temperature = 0, and each
non-edge student profile was evaluated across five independent replicates. The variance
analysis (Section 6) shows that the per-student standard deviation of accuracy across
those replicates is below 0.015 for all model–prompt combinations, with DeepSeek
consistently below 0.010. The observed forgetting effects are therefore highly
reproducible and cannot be attributed to random variation in the model's outputs. This
property holds across all three prompt strategies and both grade levels.

---

## Step 6 — Prompt Strategy Is the Primary Determinant of Quality

Ranking the nine model–prompt combinations by controllability score (Grade 4) reveals
that prompt choice matters more than model choice:

| Configuration | Controllability Score (Grade 4) |
|---|---|
| GPT-4o, `rule_based` | **0.317** |
| GPT-4o, `combined` | **0.315** |
| DeepSeek, `combined` | 0.174 |
| Claude, `combined` / `rule_based` | 0.171–0.172 |
| Claude, `few_shot` | 0.062 |
| DeepSeek, `few_shot` | −0.001 |

The two best configurations in the entire dataset are both GPT-4o configurations; the
worst are `few_shot` variants regardless of model. The skill-profile deviation metric
(RMSE, lower is better) corroborates this ranking: GPT-4o under `rule_based` and
`combined` achieves RMSE values of 0.230 and 0.235 respectively — the lowest observed
across all configurations — while `few_shot` variants consistently produce RMSE values
between 0.60 and 0.72, approaching the level of a model that ignores the instruction
entirely.

For Grade 5, the ranking shifts slightly: DeepSeek under `combined` leads with a
controllability score of 0.254, followed by GPT-4o `rule_based` (0.234) and
`combined` (0.205). DeepSeek also achieves the lowest Grade 5 RMSE (0.350), suggesting
that its skill-level targeting improves relative to GPT-4o as the number of skills
decreases.

---

## Step 7 — Analysis of the Intended vs. Observed Profile Gap

A more granular examination of the gap between the prescribed skill profile and the
model's observed behaviour reveals two distinct failure modes.

**Under-forgetting** is the dominant failure for `few_shot`. DeepSeek's forgotten-skill
accuracy under `few_shot` (q₁ ≈ q₀) indicates that the few-shot demonstrations provide
insufficient signal to override the model's learned tendency to answer correctly. The
model effectively treats the examples as stylistic guidance rather than as instructions
about which skills to suppress.

**Over-forgetting** affects `rule_based` and, to a lesser extent, `combined` configurations.
Several configurations drive forgotten-skill accuracy well below the random-chance
baseline of 0.25 — for example, GPT-4o `rule_based` Grade 4, Operations & Algebraic
Thinking, yields q₁ = 0.042. A real student who has forgotten a skill would still answer
at chance level, not systematically below it, because the LLM in this regime appears to
have learned to actively avoid what it perceives to be correct answers, which is an
unnatural behavioural artefact.

The `combined` prompt achieves the best balance, generally driving forgotten-skill
accuracy into the range 0.15–0.30 (close to or at the chance level of 0.25) while keeping
retained-skill accuracy near q₀. The gap between intended and observed profiles is
narrowest under this configuration, particularly for GPT-4o and DeepSeek.

Claude's Grade 5 results present an additional confound: Skill 2 (Number &
Operations — Fractions) has a baseline accuracy of only q₀ = 0.24 under `combined`
and q₀ = 0.09 under `rule_based`, leaving almost no room to demonstrate forgetting
beyond the chance level. This near-floor baseline limits the interpretability of
Claude's Grade 5 forgetting metrics.

---

## Step 8 — Model Comparison: Ability to Simulate Imperfect Students

Taken together, the three models occupy distinct positions in the capability space of
imperfect-student simulation.

**GPT-4o** achieves the best overall balance of target drop and retention fidelity. It
produces the highest controllability scores in Grade 4, the lowest RMSE values across
both grades, and the smallest off-target influence under both `rule_based` and `combined`
prompts. Its forgetting is both deep (accuracy reaches or approaches chance level) and
localised (retained skills are largely unaffected). These properties make it the most
reliable simulator of a prescribed skill-deficit profile.

**DeepSeek-Chat** is the most stable and reproducible model, with per-replicate variance
consistently below 0.010. Under `combined`, it achieves the best Grade 5 controllability
score (0.254) and produces near-zero off-target influence (k = 1 mean r_i = 0.017 for
Grade 4), matching GPT-4o on retention fidelity. Its primary limitation is that its
target drop under the best prompt (0.368 for Grade 4) is substantially lower than GPT-4o's
(0.695), meaning its forgotten skills do not reliably reach the chance level. Under
`few_shot`, DeepSeek fails entirely, with a controllability score of −0.001.

**Claude Sonnet** demonstrates the strongest raw target drop (up to 0.869 under
`rule_based`, Grade 4) but pairs this with the highest off-target damage of the three
models. Its `rule_based` k = 1 mean relative loss of 0.548 indicates that retained skills
are nearly as affected as forgotten ones, negating any claim of selectivity. Under
`combined`, Claude's controllability score (0.172) is comparable to DeepSeek's but
its off-target influence (0.089) is higher. The model's strong prompt sensitivity in
baseline performance further limits its suitability as a precision simulator without
additional prompt calibration.

---

## Step 9 — Baseline-Normalised Forgetting Efficiency: A Fair Cross-Model Comparison

The three models enter the forgetting task from different starting positions.  DeepSeek and
Claude achieve high baseline accuracy on most skills (q₀ ≈ 0.88–0.96), while GPT-4o's
perfect-student baseline is markedly lower on several skills (q₀ ≈ 0.55–0.72).  This
matters because the raw metric r_i = (q₀ − q₁) / q₀ does not penalise a model that starts
low: a model that begins at q₀ = 0.60 and stays at q₁ = 0.60 has r_i = 0 (no forgetting),
but a model that begins at 0.60 and drops to 0.20 has r_i = 0.67, which superficially looks
impressive even though q₁ = 0.20 is *below the random-guessing baseline* (0.25 for a
4-choice item).

### The r_max correction

For a 4-choice multiple-choice test the floor of legitimate student accuracy is the chance
level c = 0.25.  The maximum theoretically meaningful relative loss for a given skill is
therefore

> r_max  =  (q₀ − 0.25) / q₀

A model with q₀ = 0.92 (typical for DeepSeek) has r_max ≈ 0.73.  
A model with q₀ = 0.65 (typical for GPT-4o) has r_max ≈ 0.62.

We define the **Forgetting Efficiency** as

> Efficiency  =  r_i / r_max

- **Efficiency = 1.0** — the model dropped the skill exactly to the chance level.  This is the ideal outcome: the skill is fully forgotten but the model does not behave worse than a naïve guesser.
- **Efficiency < 1.0** — partial forgetting; the skill is degraded but accuracy remains above chance.
- **Efficiency > 1.0** — *over-forgetting*; accuracy has dropped below chance, meaning the model is now actively avoiding the correct answer.  This has no counterpart in human learning and is an artefact of the prompting mechanism.

### Findings after normalisation

Once efficiency is computed, the model ranking changes substantially:

| Model | Typical q₀ | Raw target drop (combined) | Efficiency | Overshoot rate |
|-------|-----------|--------------------------|------------|----------------|
| DeepSeek-Chat | 0.88–0.96 | ~0.37 | ~0.50 | low |
| GPT-4o | 0.55–0.72 | ~0.70 | >1.0 on many skills | moderate–high |
| Claude Sonnet | 0.65–0.87 | ~0.55–0.87 | >1.0 under rule_based | high |

**DeepSeek**, despite its modest raw target drop, consistently achieves efficiencies around
0.50–0.65: it uses roughly half of the room available before chance, without overshooting.
This is genuinely controlled partial forgetting.

**GPT-4o** reports the highest raw target drop, but a substantial fraction of its forgotten
skills have efficiency > 1.0, meaning q₁ < 0.25.  After normalisation GPT-4o is not more
efficient than DeepSeek; it is simply a less capable baseline model, and the prompt pushes it
below chance.  This is not selective forgetting — it is score collapse on already-weak skills.

**Claude Sonnet** under `rule_based` shows the most severe overshoot: a large proportion of
its k = 0 skills reach efficiency values of 1.5–2.0, indicating that the model is performing
substantially below chance on those skills.  Under the `combined` prompt the overshoot rate is
reduced but not eliminated.

### Implication for prompt design

The efficiency metric reveals that **none of the models currently achieve precision
forgetting** (efficiency ≈ 1.0 with low variance across skills).  The practical gap is not
about which model forgets *more* — it is about which model forgets *to the right depth*.
DeepSeek under `combined` is closest to the ideal profile: moderate, above-chance accuracy
reduction without overshooting.  Calibration of the forgetting signal intensity
(e.g., graded instruction strength, or a post-hoc accuracy constraint) is the primary
engineering target for the next phase.

---

## Summary and Answer to the Research Question

The research question is answered **affirmatively** for GPT-4o and DeepSeek under the
`combined` prompt strategy, and **conditionally affirmatively** for Claude. A prompt can
be configured to cause an LLM to exhibit meaningfully lower accuracy on a set of
designated forgotten skills while preserving, or nearly preserving, performance on the
remaining retained skills. The effect is statistically robust (variance < 0.015 across
five replicates), skill-selective (confirmed by cross-skill influence matrices), and
graded in magnitude by prompt design.

The key finding across all three models is that **no single prompt strategy dominates on
all dimensions simultaneously**. The `few_shot` prompt produces high retention fidelity
but near-zero forgetting. The `rule_based` prompt produces deep forgetting but with
collateral damage on retained skills. The `combined` prompt, which integrates both
rule-based constraints and demonstration examples, consistently achieves the best balance
and is the recommended strategy for constructing imperfect-student simulators.

The principal remaining gap between the best observed results and a theoretically perfect
simulation lies in **target precision**: current configurations either under-forget (accuracy
remains above the chance level) or over-forget (accuracy drops below chance, a behaviour
with no psychological counterpart in a human student). Closing this gap — for example,
through calibration of the forgetting signal intensity — constitutes the natural next step
for this line of research.

---

*All figures are derived from `master_experiment/experiments/exp/analysis/` using the
combined dataset covering experiments 013–015, 017 (DeepSeek-Chat, GPT-4o,
Claude Sonnet, Grades 4–5, 3 prompt strategies, temperature = 0).*
