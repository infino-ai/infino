A three-superfile FTS table (`body` column with positions, 40 rows per
superfile) written by the engine as it stood just before the term index kept
per-superfile heads, then run through a compaction-free `optimize`: its term
index is complete, with a root at layout version 3 naming slices at version 1,
and no tier objects.

The compatibility test opens it with the current reader and checks that it
answers identically — at a pruning `k`, with the old index serving ceilings and
reading no tier — before and after the current writer appends to it and the
maintenance rebuild writes the current layout, after which the tiers serve.
Regenerate only if that older layout itself must change, which it should not:
the point of the fixture is that tables indexed by it keep working.
