A three-segment FTS table (`title` column, 40 rows per segment) written by the
engine as it stood before the table-level term index existed, then run through a
compaction-free `optimize` so its manifest carries the term-stats sidecar. Its
manifest parts hold per-superfile term blooms and term ranges, its list holds
per-part bloom unions, and it references no term index.

The format-compatibility test opens it with the current reader and checks that
it answers identically before and after the current writer appends to it and
rebuilds its index. Regenerate only if the old format itself must change, which
it should not: the point of the fixture is that old tables keep working.
