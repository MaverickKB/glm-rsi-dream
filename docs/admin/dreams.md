# Dreams and schedule

Under development. Dream, schedule and consolidation actions are closed by default on the server. The dream-neurons owner must pass the pocket-OFF recall bar before the host opens development access.

## Simple

### Check availability

Open Dreams and schedule to see Under development. Continue reviewing the queue and managing raw-fact pockets. Accepted or staged facts are preparation records; they do not show knowledge moving into neurons.

### Review a development plan after opening

When the owner qualification passes and the host explicitly opens the flag, choose a nonempty staged run to prepare a manual plan. Check its waiting reasons. The API performs plan preparation; native training and qualification remain separate operator work.

## Advanced

### Inspect the server gate

Read the feature status and opening criterion. Browser settings cannot open this flag. Gated API operations return feature_unavailable: under development.

### Configure development scheduling after opening

After the host opens qualified development access, inspect time zone, approved windows, idle threshold and active requests. Readiness uses supplied context and the signed stage permission; it produces a manual plan.

## Settings and details

### Dream availability

The single host-owned dream_features_enabled flag defaults to false. The API startup option --enable-dream-features opens development access only after dream-neurons passes its pocket-OFF recall bar. GET /system reports features.dream with enabled, label, status, owner and opening_criterion. All schedule and dream endpoints return HTTP 503 feature_unavailable: under development while closed. The browser reads this state and exposes no dream or schedule action in either layer. Queue, grading, rules and raw-fact pockets remain usable.

### Schedule enabled

Development setting, available after the server gate opens. Enables eligibility checks. The default is false. Enabling requires at least one approved window and keeps training execution under the manual operator workflow.

### Timezone

Development setting, available after the server gate opens. An IANA timezone such as America/New_York, the default. Approved hours are interpreted in that zone, including its daylight-saving changes.

### Approved days

Development setting, available after the server gate opens. Choose the days on which a window starts. API values are Monday 0 through Sunday 6. An overnight window continues into the following day.

### Approved hours

Development setting, available after the server gate opens. Start and end use local 24-hour HH:MM. Start is included and end is excluded. An earlier end crosses midnight. Equal start and end are rejected. Up to 30 windows are supported.

### Minimum idle time

Development setting, available after the server gate opens. Required idle seconds before eligibility, from 0 to 604800. The default is 900, or 15 minutes.

### Maximum active requests

Development setting, available after the server gate opens. The largest supplied active-request count permitted for eligibility, from 0 to 1000. The default 0 requires an empty request count.

### Staged run

Development setting, available after the server gate opens. Run_id selects an existing nonempty sealed staging manifest. A dream job receives its own new ID and records who approved the plan.

### Earliest time

Development setting, available after the server gate opens. A timezone-aware timestamp. Readiness stays not_due until this time; the schedule window and idle checks must also pass.

### Readiness inputs

Development setting, available after the server gate opens. At is a timezone-aware check time. Idle_seconds and active_requests are nonnegative counts supplied by the admin. These values describe the check; the API does not measure live idle state.

### Readiness result

Development setting, available after the server gate opens. Eligible means the supplied context, approved window, stage permission and nonempty staging passed. Next_action is human_confirm. The result includes executes_training:false and each waiting reason.

[All pages](README.md)
