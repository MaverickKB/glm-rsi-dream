# Labeled funnel development set

`funnel_gold.json` has 168 authored candidate facts in 24 mixed source documents.
There are 24 candidates in each category: known facts, new durable knowledge,
personal ephemera, continuity decisions, synthetic secrets, updates, and duplicates.
Every candidate has literal source evidence, an extraction label, a retention
verdict, a route, and relevant prior/update/duplicate metadata.

The identities, local records, identifiers and credentials are fictional. Secret
examples carry TEST markers. All 168 claims are extraction targets. Secret,
already-known and unchanged duplicate claims have retention-reject labels.
Ephemera and attributed continuity decisions have baseline memory labels.
General definitions route to weights; local durable facts and updates route to
named pockets. These destinations are an explicit development hypothesis.

Each document includes a repeated fact, a question, an unverified guess, and
periodic untrusted instructions. Repeated source occurrences have one gold fact.
Questions, guesses, instructions and the fictional-exercise header are not gold
claims. Extra outputs count as false positives against the annotated targets;
that does not establish that every extra output is a hallucination.

The extractor sees only source text in ordinal TXT, Markdown, HTML or DOCX files.
Gold labels, catalog priors and routing metadata are held outside collection.
The closed-book prior receives only its generated question. A later comparison
receives the candidate and prior answer. Post-collection scoring counts exact
text matches deterministically. For remaining claims, GLM compares each candidate
with remaining gold text and returns a full-fact boolean and reason. Code owns
one-to-one accounting; GLM receives the two texts without category or route labels.
Matching requires the complete atomic entity, value, ownership, time, polarity,
scope and update relation. Missing claims are FN; extra, duplicate or bundled
outputs are FP. Precision is TP / all outputs; recall is TP / 168 targets.
Per-category recall uses gold support. Unmatched outputs have no true category.

`funnel_sources_gold.json` is the same corpus in the source/expected schema used
by the full-funnel evaluator. Its blind novelty labels concern the model's own
weights: general known facts are known; private fictional facts are new.
`catalog_novelty` preserves the supplied prior used by the flat grading fixtures.
An explicit external prior record can show an update or duplicate without that
private record being present in the model's closed-book knowledge. These are
separate measurements. Gold duplicate rejection assumes the stated prior record;
gold update acceptance assumes the explicit correction in the source.

This is a sealed development set. The same GLM performs extraction and separate
semantic alignment, so the automated score has correlated-judge limitations.
Preserve raw calls, alignment reasons, the input seal and the exact source hash.
Use a new run root for changed policy, code, gold or source documents.

The tested hypothesis is complete, source-supported atomic discovery before
retention decisions. High output count alone cannot support that hypothesis.
Missing updates, losing ownership or dates, producing unsupported claims, or
merging independent claims falsifies complete discovery even with valid JSON.
If a product intentionally drops known facts or duplicates before its queue,
this gold counts those as discovery misses while its final rejection may still
be correct. Compare the category recalls with full-funnel retention metrics.
Synthetic identities, repeated document structure and reuse during tuning limit
generalization. This set does not measure learned recall after staging.
