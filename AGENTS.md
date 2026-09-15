# AI_2 Agent Rules

## Vast.ai instance lifecycle

- Run a live `vastai show instances --raw` inventory immediately before creating any instance. Do not rely on an earlier receipt.
- Never create a replacement merely because an existing instance is inconvenient, stopped, loading, or owned by an unknown lane. First identify its owner and either reuse it, coordinate with its owner, or close it out.
- An agent may own at most one Vast.ai instance at a time. Every instance label must identify the project, task, owner/lane, and date, and its ID must be recorded locally as soon as it is created.
- If another agent is actively using an instance, do not stop or destroy it. A separate instance is allowed only when the user explicitly requested independent testing and the new agent still owns no instance.
- When the owning agent no longer needs its instance, it must immediately retrieve and verify every needed artifact, then destroy the instance. Stopping it is not closeout because retained disks continue to cost money. The user has given standing authorization for an agent to destroy its own unused instance after artifact verification.
- Before ending any GPU task, run a fresh live inventory and report every remaining instance by ID, label, state, owner, purpose, and hourly cost. Zero unexplained instances is the completion condition.
