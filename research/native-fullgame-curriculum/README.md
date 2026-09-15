# Native Full-Game Curriculum MVP

Task 1 locks the runtime action table before any native behavior edits. The table is necessary, but Task 2 must still prove Python/native trace parity for all ten actions.

## Stale Action Labels

Use this audit before relying on older docs, validators, dashboards, or reports for full-game action semantics:

```bash
rg -n "5=dodge|6=pickup|9=throw|\\[5,\\s*8,\\s*9\\].*tool|throw.*index 9|index 9.*throw" .
```

Expected result: any hits are either fixed or explicitly listed as stale historical labels that cannot drive runtime behavior.
