## Research Grounding

This project is built on current research in survival-based credit
scoring, explainable AI, and reject inference -- and goes beyond it in
several specific, tested ways rather than simply applying it.

**Adopted directly.** The core explanation method is a time-dependent
technique designed specifically for survival models, used in place of
standard feature-attribution methods that are known to misrepresent
skewed, time-to-event outcomes. The reject-inference approach follows a
diagnose-before-correcting design: a measurable test decides whether
selection bias is actually present before any correction is applied,
rather than applying a correction by default.

**Tested, not assumed.** A documented concern in the explainability
literature -- that standard feature-attribution methods distort survival
outcomes -- was tested directly on this project's own data rather than
taken on faith. Result: the two methods agreed more closely with each
other than either agreed with itself across repeated runs, meaning the
concern did not reproduce here at the level of which reasons get
reported. Separately, a documented warning that reject-inference
corrections don't reliably improve model accuracy was also tested
directly: bias was detected, a correction was applied, and accuracy did
not improve, exactly as the warning predicts. Both results are reported
regardless of which way they came out.

**Extended beyond the literature.** Existing reject-inference research
asks whether a correction changes a model's accuracy. This project asks
a further question: does it change *why* the model decides what it
decides? Re-running the explanation step before and after correction
showed the overall ranking of important factors stayed stable, but two
specific features shifted substantially in weight even while remaining
outside the top ranks -- a finding a pure accuracy comparison would
never surface.

**A real, measured fair-lending result.** Applicant state/geography
ranked among the strongest features by one importance measure, yet
removing it entirely cost a negligible, measured amount of accuracy.
That trade was tested, not assumed, and the model used by this system
excludes geography as a result.

**A rigorous negative result.** A much faster alternative explanation
method was tested against a pre-registered accuracy bar before being
considered for production use. It failed the bar on one required
criterion. It was kept only as an internal screening tool and is never
used for anything disclosed to an applicant -- a negative result,
reported as one, not hidden.

**A defect found and fixed.** An earlier version of the model was
missing two legitimate, pre-decision applicant features due to a
specification error, not a deliberate design choice. Fixing it
measurably improved accuracy and enabled a model that uses no
information from a lender's own pricing decision and no geography at
all, while matching the accuracy of the model that used both.

**Governance the research literature doesn't need, but a deployed
system does.** None of the techniques above address what happens when a
model is incomplete or wrong in production. This project adds a model
registry with automatically-checked approval rules, mandatory
post-run validation on every scoring run, and a strict separation
between what an applicant is legally told and what stays in an internal
fair-lending review record -- closing a real gap this project found in
its own early output.

Full evidence, every number, and every negative result are documented in
`FINDINGS.md`.
