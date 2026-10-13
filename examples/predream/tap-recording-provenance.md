# Recorded tap regression input

`tests/predream/fixtures/tap_recorded.jsonl` preserves the first nine GLM
request/response records from the sibling LOOP2-extract fixture-01 capture on
2026-10-12. They cover one extraction request containing four ordinary
arithmetic/geography claims, four blind prior questions and four comparisons.
The source is the existing fictional regression campaign, with no credentials
or private facts. Original messages and responses are preserved byte-for-byte
per JSONL line.

The replay client matches the protocol and the complete exchange after the
system message. Current comparison instructions add a reason request to the
historical rubric. The recorded comparisons contain the original valid
novelty/agreement fields. Playback checks queue wiring against those responses;
it does not measure how a fresh GLM would answer the newer rubric or ordinary
live traffic.
