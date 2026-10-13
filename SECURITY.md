# Security

The written security review target is v0.0.1. The package version is 0.0.2.
That package revision was not a new security review. This experimental package starts
temporary Docker containers and uses SSH on a two-node Spark pair. CPU checks
cover local contracts. Later installed-source serving and rollback passed
bounded qualification; this public checkout's exact native CLI remains a
verification target. See README.md for the tested implementation and limits.

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
