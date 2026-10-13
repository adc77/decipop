# Proposed next experiment: decisions about the right entity

Status: proposal after V4; no dataset freeze, GPU submission, or training run.

## Question

Does the model follow the queried entity when the same facts and rules support different answers about different entities?

V4 improved unsupported-query UNKNOWN accuracy to 81.5%, but its generator still distinguished one target with a selector predicate. A model could exploit that cue without applying the rules independently to each entity. Wording and single-branch transfer also remain weak. Another training run would be premature until we measure this gap.

## A simple example

Facts: Hana is red and kind. Evan is red and not kind. Mina is red.

Rules: If someone is red and kind, they are verified. If someone is red and not kind, they are not verified.

| Statement about the same world | Oracle answer |
| --- | --- |
| Hana is verified. | TRUE |
| Evan is verified. | FALSE |
| Mina is verified. | UNKNOWN |

Every entity shares the first premise. There is no unique selector identifying one special target. Changing Hana's support should change Hana's answer while preserving the answers about Evan and Mina.

This example illustrates the design; it is not a selected model result or the final dataset specification.

## Work in order

1. Build a separate generator and audit it on CPU. Generate several independently answerable entities in every world, with TRUE, FALSE, and UNKNOWN answers about the same predicate. Reject contradictions and inputs exceeding the models' context limit. Split by parent world, including all renamings and interventions in the same partition. Exclude canonical worlds recorded in earlier studies.
2. Add matched checks: change only the queried entity; consistently permute names; reorder facts and queries; insert irrelevant facts; remove or reverse support for one entity. Check oracle labels and ensure other entities' answers are preserved where intended. Balance names, answer positions, polarity, and label exposure. Shared cues can still exist, so record generator limitations explicitly.
3. Freeze the diagnostic and analysis before scoring. Compare pinned V3 treatment-final, V4 control, and V4 mixed-data heads. Use each release's saved temperature without fitting on diagnostic labels. Report overall and per-label accuracy, complete-world accuracy, intervention correctness, unaffected-entity consistency, and confident errors. Treat the parent world as the sampling unit. This measures retained fixed models, not training-seed uncertainty.
4. Run a bounded inference-only Modal batch after checking current credits, timing a small pilot, and recording a submission ceiling. Retain raw logits, oracle traces, identities, and failed calls. Verify downloads before deleting the owned temporary Volume.
5. Review the failures before choosing a second experiment. If query/entity consistency fails, compare matched additional training on these worlds. If it passes but wording or rule structure fails, target that specific transfer gap. Consider architectural changes only with a concrete hypothesis and a matched baseline.

No new credentials are needed for preparation: existing GitHub, Hugging Face, and Modal authentication worked during the completed studies and publication. Budget and access must be rechecked before a new GPU submission. The next experiment's sample count, seed, statistical endpoints, and cost cap will be specified in its own reviewed protocol; this proposal does not freeze them.
