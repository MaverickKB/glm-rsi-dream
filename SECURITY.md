# Security

The current review target is v0.0.1-preview. This experimental package starts
temporary Docker containers and uses SSH on a two-node Spark pair. CPU checks
cover local contracts; native serving and rollback acceptance are in progress.

Report vulnerabilities through GitHub private vulnerability reporting: open
this repository's **Security** tab, select **Report a vulnerability**, and
include the affected version, a minimal fictional reproduction, and impact.
The publisher must enable private vulnerability reporting before publication.
If the option is unavailable, open an issue requesting a private reporting
channel, without sensitive details.

Keep credentials, private prompts, tensor captures, internal addresses, and
host configuration out of public issues. Redact these from logs and replace
real facts with fictional examples. The maintainer will coordinate investigation
and disclosure through the private advisory.
