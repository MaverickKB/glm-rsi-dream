# Queue

Review one claim at a time. Read its evidence and proposed destination before deciding what should move forward.

## Simple

### Learn prepared knowledge

Choose Learn this, paste a note or upload a UTF-8 text or JSON file, then select Add for review. GLM extracts text into facts, checks safety and grounding, and proposes a destination. Open Review fact to approve or reject the proposal under your current autonomy policy.

### Review a proposal

Find a pending item and open its details. Compare the claim with the source evidence, then read the model's reason. Approve a suitable proposal, reject an unsuitable one with a reason, or use Hold when you need more information.

### Correct the evidence

Edit the claim and literal evidence together. Saving creates a pending replacement and holds the original. The replacement needs fresh model review and admin approval.

## Advanced

### Submit a prepared pack

Choose Learn this to set an explicit input format and source reference. Upload a pack or paste its JSON. Retrying the same reference and content reuses the queue IDs. If a grade fails, open the pending fact and choose Grade fact; later facts remain available for their own review.

### Inspect the trail

Use filters and the raw detail view to inspect proposals, approvals, update links and receipts. Select rows for the bulk actions offered by the portal; review any failed row before retrying.

### Change completed knowledge

Prepare a linked update for a staged, absorbed or learned item. Keep the previous receipt intact while the replacement is reviewed. A learned receipt records the consolidation artifact; an accepted or staged row still awaits the relevant handoff.

## Settings and details

### Claim text

A self-contained statement, 1 to 16000 characters. Editing it requires the literal evidence at the same time and creates a new candidate identity.

### Literal evidence

The source span that supports the claim, 1 to 16000 characters. Supply it together with claim text when editing knowledge. Credential-shaped input is rejected before storage.

### Hold

Stops this candidate from approval or handoff while you review it. Editing a review field clears applied approval. Releasing a hold leaves the proposal awaiting a fresh decision.

### Review note

Optional admin context, up to 2000 characters. Saving a review edit returns the candidate to pending review.

### Model proposal

The model's accept or reject verdict, destination, reason and score are review evidence. Approve applies that proposal with an admin receipt. An explicit admin rejection records a separate veto and preserves the model judgment. Hold keeps an item pending.

### Queue status

Pending awaits review; accepted or rejected records an applied decision; staged holds raw reusable knowledge; absorbed records a memory handoff; learned requires a provenance receipt. Staging alone records preparation.

### Search and selection

Search and filters narrow the visible table. Row selection chooses candidates for the actions displayed by the portal. Review their IDs and individual outcomes before a bulk save.

### Revision

The API binds an edit to the version you read. Queue revisions are SHA-256 strings; other item revisions are integers. A 409 means the item changed. Reload it and reconsider your edit.

### Rejection reason

Required nonempty text for an explicit admin veto. The API records the admin identity and time while preserving the model proposal. Completed handoffs require a linked update.

### Prepared content

Paste text or upload a UTF-8 .txt, .md or .json file. The API accepts up to 1000000 characters; the portal file limit is 1 MB. JSON has exactly a facts array with 1 to 2000 objects containing statement and optional prompt_evidence. Each statement and evidence is nonempty and at most 16000 characters. Omitted evidence uses the supplied statement as the user assertion. Extra fields and duplicate statements are rejected. Example: {"facts":[{"statement":"Fictional Lumen has three sample bays.","prompt_evidence":"The Lumen specification lists three sample bays."}]}. Credential-shaped input is rejected before storage or model access.

### Input format

Text asks GLM to discover atomic facts and verify literal evidence spans. JSON declares the facts directly. Manual intake records unmeasured novelty and skips blind novelty probes; GLM safety, grounding, relationship and destination grading still apply.

### Source reference

A nonempty label, up to 512 characters, identifying your prepared source. Simple paste uses manual:portal; an upload uses manual:file: plus its filename. Advanced can set your own reference. The queue adds manual intent and the content hash. Keep the same reference for retry, and use a distinct reference for a distinct source.

### Prepared grading and retry

Add for review queues the submitted version and grades its facts in order. A failed model review leaves the fact pending, shows an error and stops that pass. Grade fact retries an ungraded manual candidate against its current queue revision. Existing proposals and decided items keep their judgments on replay. Manual autonomy waits for explicit admin approval; supervised and Automatic use their signed permission bounds. Grading prepares knowledge for the normal memory or staging handoff.

[All pages](README.md)
