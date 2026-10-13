# Autonomy

Choose how facts are saved. Select Manual, Supervised or Automatic to apply that level.

## Simple

### Keep review in your hands

Select Manual. Review pending proposals in Queue and approve or reject each fact.

### Allow automatic saving

Select Supervised or Automatic to apply the level. If the option says Needs your approval to turn on, choose its Approve button. Supervised asks you to review uncertain facts and changes.

## Advanced

### Use Automatic deliberately

Select Automatic to save facts after the usual checks. Open Access details for scope and expiry. The controller continues to enforce safety, candidate review and handoff evidence.

### Change control settings

Open Edit settings to choose destinations and the confidence needed. Future decisions use the saved settings. Completed staged, absorbed and learned facts keep their sealed state; pending and accepted proposals reopen for review when settings change.

## Settings and details

### Autonomy level

Manual requires explicit candidate approval. Supervised permits grant-backed decisions within route and score limits, holding conflicts and updates for review. Automatic permits automatic decisions within its signed grant. The default is manual.

### Automatic routes

Supervised destinations allowed for automatic approval: weights, pocket and memory. The default is memory only. The model still proposes the destination.

### Minimum score

Supervised approval requires a model score at or above this value, from 0 to 1. The default is 0.9. The score is a model judgment; inspect its reason.

### Permission file

A host-admin signed grant binds the storage root, execution mode, autonomy, allowed actions and expiry. The host provisions its private path. An explicit null policy value closes retention; omitting the field allows the host environment setting.

### Signed authority

Available means the controller verified a grant for the checked action. Inspect the admin ID and expiry. Separate bearer access lets the browser reach the API; action authority comes from this grant.

[All pages](README.md)
