# Git history preservation

Original repositories and dirty working trees were not reset, committed over or pushed to their upstream remotes. Bundles of every original repository are stored in the sibling `_local-history/` directory. These bundles are local backups, not GitHub upload content. Modified working-tree code is preserved in the file manifest separately from original commits.

The new CIL main commit descends from the clean upstream CDS commit. Safe upstream RevisitingCIL and BeyondObjects histories are retained under `history/` tags. Both CDS-derived local repositories share the same upstream commit. Their separate original bundles preserve their refs; their distinct working changes appear in their own source folders.

The early diagnostic history contains personal home paths in historical configs/scripts. Its bundle and original `.git` are preserved locally but deliberately excluded from GitHub. Importing it into a published branch would leak those paths; rewriting it would break original commit identities. This documented local-only preservation satisfies both constraints as far as possible.

`provenance/git_history.csv` records original remotes, heads, dirty states and bundle locations. The CLIP source directories had no original Git repository in the inspected workspace, so the CLIP archive has a new initial commit. Normal publishing pushes only main and the safe history tags; no force push or branch deletion is needed.
