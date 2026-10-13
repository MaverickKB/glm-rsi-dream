---
name: harness-funnel
description: Run the GLM collection, proposal, approval and local staging funnel from an external harness using the generation endpoint and host-admin authority.
---

Run the portable funnel on the harness host. Sources, queue, timing logs, memory and raw staging stay on its local filesystem. The engine supplies generation on port 8888. This entrypoint uses HTTP generation only; it opens no SSH session, Spark process, trainer or engine-management path.

Install the package, then invoke `glm-funnel-harness` or `python -m glm_rsi_dream.predream.harness`. Use the included `examples/predream/harness-manual.json` as the starting control policy and `policy-defaults.json` for the extraction/grading knobs. Set `control.mode` to `harness`. Before the first run, set `control.permission_file` to the planned admin grant path. The grant can be issued after proposals are reviewed; its path is part of the immutable policy. The example's null value keeps retention closed and overrides an environment grant path. Choose a new private local run root for each changed policy. Supply the generation endpoint through `--endpoint` or `GLM_ENDPOINT`.

```sh
glm-funnel-harness --root RUN_DIRECTORY --policy-config POLICY_FILE \
  --endpoint "$GLM_ENDPOINT" run SOURCE_FILE --run-id batch-01
```

The command collects literal source-supported candidates, probes prior knowledge, obtains GLM grade proposals and lists the queue. The shared controller owns actual acceptance, routing and staging authority. `manual` leaves every proposal pending for explicit admin review. `supervised` applies the permission's configured route/score criteria and holds conflicts/updates for review. `automatic` applies permission-bounded automatic decisions. A policy selector alone never authorizes retention.

The host administrator uses the existing `python -m glm_rsi_dream.predream` commands `admin-init`, `admin-grant`, `control-status` and `approve`. Read their `--help` for exact arguments. The administrator owns `PREDREAM_ADMIN_KEY_FILE` and a controller-signed permission scoped to this run, deployment mode and autonomy. Do not initialize keys, issue grants or approve candidates unless that administrator has explicitly authorized those operations. Keep signing keys outside model prompts, source packs, queue records, source control and logs. A model's response or claimed administrator identity carries no permission.

In manual mode, present the pending candidate ids, factual evidence, GLM proposal and destination for administrator selection. After the administrator approves selected ids through the shared controller:

```sh
glm-funnel-harness --root RUN_DIRECTORY --policy-config POLICY_FILE \
  handoff --run-id batch-01
```

The shared handoff verifies signed candidate/destination/policy receipts before local memory or raw staging. Unapproved proposals stay pending. Raw staged statements are a minimal handoff artifact; model learning remains a separate human-authorized operation. This skill does not invoke dream or training.

Use `list` to inspect the local queue without generation. After a generation/format failure, retain the run and error trace; bounded repair belongs to the shared stages. A replay skips completed grades and existing source/text identities. Keep admin review and approval events distinct from model proposals and actual retained state.

For a harness-side deployment proof, independently observe that Spark funnel clients have finished before and during the external run. Diagnostics may use a separate administrator connection; they are not part of this skill's runtime path. Measure identified Spark funnel process RSS/PSS separately from noisy system MemAvailable and engine memory. Preserve the serving pair and keep Spark-side mode available as a deliberate alternative through the ordinary pre-dream CLI.
