# Rules

Choose what is worth keeping. Start with a small rule, then check it against a sample before relying on it.

## Simple

### Make a rule

Create a rule or choose a companion or persona preset. Give it a useful name, choose a destination and save. Open its details to check who and what it covers.

### Try it first

Use the rule check with a sample user, source and text. Review the selected match. The check evaluates proposal settings; collection remains off.

## Advanced

### Narrow the match

Open the rule editor to set exclusions, source kinds, literal patterns and precedence. Disable a rule when you want to keep it for later.

### Tune the pipeline

Open collection, novelty and grading settings. Export your policy for a new run root. The active experiment keeps its original policy and source snapshot. Keep the defaults until you have a specific hypothesis to test.

## Settings and details

### Rule name

A readable label, 1 to 100 characters. Choose words that tell you what the rule covers.

### Rule enabled

Enabled rules participate in rule checks. Disabled rules stay saved. The default is enabled.

### Domain

General matches every checked domain. Companion and persona match their own domain. Choose the scope that fits the rule.

### Action

Propose records an intended retention policy; ignore records an intended exclusion. Rule checks return the selected policy for review.

### Destination and pocket

Weights means reusable general knowledge; pocket means reusable topic knowledge; memory means personal or transient information. Pocket requires a safe topic ID. The other routes require a null pocket. A rule expresses a proposal policy; model review and admin approval still govern candidate handoff.

### Priority

Lower numbers are checked first, from 0 to 1000. The default is 100. Ties use the rule ID in alphabetical order.

### Included users

A list limits the rule to those user IDs. An empty list includes every user who passes the exclusion check.

### Excluded users

These user IDs never match this rule, even when also included. Exclusions are checked before other matching conditions.

### Source kinds

Choose one or more of prompt, conversation and manual. The default includes all three. This filters a rule check by the source kind supplied to it.

### Literal patterns

Any listed phrase may match the sample text, ignoring letter case. These are literal substrings. An empty list adds no text restriction.

### Rule notes

An optional explanation for the next reviewer, up to 2000 characters. Notes stay with the rule.

### Preset

Companion and persona provide editable starter rules. Inspect the resulting destination, topic and match scope before saving further changes.

### Rule check inputs

Supply the exact user ID, domain, source kind and sample text. The result lists matching rules and the first selected rule. It reports collection_enabled:false.

### Policy identity

Schema version is 1. Name is a nonempty experiment label. Optional _comments are saved metadata and stay outside model prompts. Every changed policy, including comments, uses a fresh run root.

### Collection instructions

Nonempty instructions for extracting atomic source claims. The literal evidence and answer-free question checks still apply. Keep the full instruction text in the focused editor.

### Novelty strategy

Closed_book, the default, asks an answer-free question before comparing the prior answer with the claim. Claim_assessment shows the claim while asking about prior knowledge. That setting requires samples 1, temperature null and threshold 1.

### Prior samples

A positive integer, default 1. Closed-book novelty makes this many prior requests for each candidate. More samples add requests and latency.

### Prior temperature

Null uses the client default of 0. A finite nonnegative value controls closed-book prior sampling.

### Agreement threshold

Greater than 0 and at most 1, default 1. For multiple prior answers, the model is instructed to use this supporting or contradicting fraction when judging novelty.

### Prior instructions

Nonempty closed-book instructions. The user input contains only the neutral probe question, so the prior request receives no proposed answer.

### Comparison instructions

Nonempty instructions for comparing the claim with the independent prior answer. The runner adds a reason request and, for multiple samples, the threshold guidance. Novelty and agreement output checks remain enforced.

### Assessment instructions

Nonempty instructions used by claim_assessment. They request a prior answer, agreement, novelty and reason while the claim is visible.

### Grading strategy

Evidence_first, the default, checks grounding, lifetime, novelty, destination and retention. Custom uses your replacement rubric. The model proposes the verdict and route in either case.

### Custom grading instructions

Required nonempty text for custom grading. Keep null with evidence_first. Custom instructions also own routing; the output protocol is appended separately.

### Routing instructions

With evidence_first, null uses the built-in destination rubric; nonempty text replaces that hypothesis. Keep null with custom grading.

### Grading attempts

An integer from 1 to 3, default 3, for obtaining a valid model grade. Exhaustion records an error and leaves the item pending for a deliberate next attempt.

### Next-run policy

Save the full version-1 policy as a next-run draft. The active run retains its immutable policy. Export the draft and use it with a fresh run root to test the change.

[All pages](README.md)
