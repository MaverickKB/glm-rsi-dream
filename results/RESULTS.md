# Source experiment results

Status: **v0.0.1-preview**. The packaged loop has not yet been run end-to-end on
Spark as packaged. Ordinary-chat serving and rollback acceptance are in progress.
These aggregate source results are separate from package acceptance.

## Round 3

The source experiment fitted 96 learned attention-memory rows on all 11
full-attention layers:

| Arm | Grounded held-out answers |
| --- | --- |
| Learned attention memory | 34/36 |
| Baseline | 1/36 |
| Full in-context carrier | 35/36 |

Development was 12/12. Controls were 30/30 byte-exact. This is a small source
set, now consumed, and provides no broad-generalization claim.

## Self-run RSI smoke: plan4

The model generated fictional facts, wrote the questions, and graded its own
answers. Learned memory scored 18/18 held-out; baseline scored 0/18; development
scored 6/6. All 30/30 tagged controls were exact. The model decided **KEEP**.

Caveats: tiny sets, self-grading bias, and tagged controls. These controls do not
establish natural-language routing. The ordinary-chat serving test is pending.
The packaged example is a different fictional fixture and has no reported score.

## Packaged acceptance

CPU tests cover local gradient, overlay, dataset, grading, and recovery
contracts. Complete native execution, fresh provisioning, kept-memory reuse,
and ordinary-chat serving/rollback remain in progress. CPU success does not
establish those live outcomes.

Only aggregate numbers are retained in [source-results.json](source-results.json).
Private documents, replies, captures, run receipts, and private-data hashes are
excluded. Scores are supplied source-experiment results, not re-measured here.
