# Problem

Open-ended preference evaluation can be decomposed into checklist questions, but
evaluating every question in a large frozen bank is expensive and may include
irrelevant criteria for a given context. The working hypothesis is that a
selector can choose a small, context-specific subset of questions that preserves
the judging signal while reducing the number of questions asked.

The current bottleneck is aligning the selector with human-relevance questions:
the selector must retrieve the questions humans would consider relevant, and
the downstream dynamic evaluator must aggregate their answers without inflating
ties or hiding useful margins.
