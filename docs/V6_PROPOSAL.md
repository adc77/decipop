# Proposed V6: learn decisions for several entities

Status: proposal after the completed V5 diagnostic. No dataset freeze, GPU submission, training, or model release.

## What we learned

V5 gave each retained model the same world and asked about three different people. Every base world required one TRUE, one FALSE, and one UNKNOWN answer. All three models scored 0/480 completely correct worlds. The V4 mixed model answered UNKNOWN on 860/864 main-suite questions, including 567 wrong answers with at least 90% confidence after its saved temperature.

This establishes a behavioral failure on a new distribution. It does not identify an internal mechanism. V4's earlier improvements remain valid on their tested distribution. The next useful question is whether training on independently answerable entities changes this behavior without losing the earlier gains.

## Matched experiment

Start both arms from the same pinned V4 mixed bundle, with fresh matched optimizers and training schedules. Control continues the existing V4 mixed curriculum on fresh families. Treatment replaces half its presentations with multi-entity examples; the other half preserves the existing curriculum. Match total question presentations, optimizer updates, label exposure, checkpoint schedule, and model/context settings. Measure token exposure separately because longer multi-entity inputs can change it.

This comparison tests a curriculum change. It does not isolate entity count from every other distribution change. Use one matched seed pair for the first bounded study; claim results for those trained models. A second seed pair requires its own budget check and is needed before asserting reproducible training gains.

## Data requirements

- Generate worlds with three to five entities that share rules but have independent support. Vary the number of TRUE, FALSE and UNKNOWN answers per world; do not require all three labels to co-occur. Balance labels over presentations rather than inserting a special target marker.
- Keep entity names, premise positions, query polarity and explicit fact counts balanced across labels. Record remaining filler, missing-premise and rule-template cues. Include conjunction rules and signed chains with positive and negative support.
- Query every entity, with randomized presentation order. Include local support removal, reversal and repair, plus entity permutations. Keep every world and its transformations in one partition.
- Exclude recorded V1–V5 canonical worlds. Use separate fresh training, development, temperature-fitting, checkpoint-selection, final-calibration and confirmatory-test families. V5 remains historical evidence; it cannot serve as the new confirmatory test.
- Audit serialized prompts against the symbolic solver and synchronous closure, reject contradictions, check intervention locality, and reject overlength inputs before training. The two solvers share parsing/grounding utilities; record that limitation.

## Selection and evaluation

Retain scheduled checkpoints for both arms. Fit temperatures on a dedicated partition and select by calibrated NLL on a different partition, following the lesson from V2–V3. Refit selected-model temperatures on the separate final-calibration partition before opening final tests. Keep raw and calibrated metrics side by side.

The proposed primary endpoint is treatment-minus-control complete-world accuracy on fresh multi-entity conjunction families, bootstrapped by parent world. Freeze the exact sample counts, seed, selector, intervals and any engineering gates before execution. Complete-world comparisons between entity counts are secondary because larger worlds contain more questions.

Report per-label recall, confusion matrices, entity-query pair correctness, intervention correctness, unaffected-entity consistency, calibration, and highly confident errors. Include fresh tests of V4 original-query and changed-premise behavior, plus deeper, wording and signed-chain transfer. Report descriptive secondary comparisons without treating them as independent confirmatory wins. Retain a preselected example even if it fails.

## Execution and decision

Preparation uses existing credentials and local CPU checks. Before any GPU call, recheck Modal prices and billing, freeze a call ledger, and run a bounded timing pilot. Proposed allocation: at most $4 compute including failed-call/recovery reservations, plus $0.50 storage/egress reserve, within the remaining original $30 budget. This is an allocation proposal, not a runtime estimate or submission authorization. Reduce scope before submission if timing cannot fit it.

Retain every selected candidate, raw logits, oracle records, costs and failure logs. Verify local retention and private pinned model downloads before deleting owned working storage. Publication is a separate owner-authorized action.

If matched multi-entity training improves fresh-world correctness while preserving earlier behavior, replicate and broaden generators before making a general decision-model claim. If it fails, inspect optimization and simple memorization gates first. A later architecture experiment needs an explicit hypothesis, such as entity-conditioned representations, and a matched baseline; changing architecture alone does not answer the V5 failure.
