A three-segment FTS table (`title` column, 40 rows per segment) written by the
engine as it stood before the table-level term index existed, then run through a
compaction-free `optimize`. Its manifest parts hold per-superfile term blooms and
term ranges, its list holds per-part bloom unions, and it references no term
index. Its manifest also names a file under `term-stats/` that the current engine
does not read; gc removes it.

The format-compatibility tests open it with the current reader and check that it
answers identically before and after the current writer appends to it and
rebuilds its index, and that `optimize` plus `gc` leave no `term-stats/` file.
Regenerate only if the old format itself must change, which it should not: the
point of the fixture is that old tables keep working.
