# Hypothesis Verification: selected questions are relevant but not discriminative
- samples: 1073
- ties: 424
- non_ties: 649
- tie all-zero selected-question samples: 294 / 424 = 0.693
- ties: selected rows=5605, nonzero row rate=0.0339, mean nonzero weight mass=0.0094, median nonzero weight mass=0.0000, mean zero weight mass=0.9906, same-label pair rate=0.9561
- non_ties: selected rows=8519, nonzero row rate=0.5102, mean nonzero weight mass=0.5472, median nonzero weight mass=0.5087, mean zero weight mass=0.4528, same-label pair rate=0.4626

## Top tie qids
- 52: selected=401, nonzero_rate=0.0125, dim=relevance_instruction_following, sub=Task Adherence
  - Does the response perform the task the user actually asked for, producing the type of output requested rather than a different kind of answer?
- 26: selected=396, nonzero_rate=0.0682, dim=correctness_and_completeness, sub=Factual Accuracy
  - Are the factual claims in the response accurate and free from errors that would mislead the user or undermine the usefulness of the answer?
- 47: selected=392, nonzero_rate=0.0051, dim=helpfulness_and_usefulness, sub=Relevance to User
  - Does the response focus on what the user actually needs to know, rather than providing information that is related to the topic but not useful for their specifi
- 22: selected=392, nonzero_rate=0.0102, dim=correctness_and_completeness, sub=Coverage Adequacy
  - Does the response address all the main aspects of the user's question without omitting key information that the user would need to consider the answer complete?
- 3: selected=388, nonzero_rate=0.0180, dim=clarity_and_communication, sub=Explanation Sufficiency
  - Does the response present enough detail to be useful without overexplaining straightforward points?
- 50: selected=384, nonzero_rate=0.0000, dim=helpfulness_and_usefulness, sub=Relevance to User
  - Does the response address the main question or request the user asked, rather than shifting to a different issue?
- 51: selected=383, nonzero_rate=0.0131, dim=relevance_instruction_following, sub=Task Adherence
  - Does the response cover each explicit request or sub-task in the user's prompt rather than answering only part of it?
- 48: selected=376, nonzero_rate=0.0000, dim=helpfulness_and_usefulness, sub=Relevance to User
  - Does the response prioritize information that directly addresses the user's stated goal or problem?
- 53: selected=332, nonzero_rate=0.0120, dim=relevance_instruction_following, sub=Task Adherence
  - Does the response deliver the specific artifact or action the user requested, for example an explanation, summary, translation, code snippet, or list, rather th
- 4: selected=325, nonzero_rate=0.0892, dim=clarity_and_communication, sub=Information Efficiency
  - Does the response convey its content without unnecessary repetition, filler, or tangential material that dilutes the useful information?

## Most collapsed qids
- 55: tie_n=49, tie_nz=0.0204, non_tie_nz=0.6889, delta=0.6685, dim=relevance_instruction_following, sub=Task Adherence
- 54: tie_n=291, tie_nz=0.0034, non_tie_nz=0.5938, delta=0.5903, dim=relevance_instruction_following, sub=Task Adherence
- 22: tie_n=392, tie_nz=0.0102, non_tie_nz=0.5945, delta=0.5843, dim=correctness_and_completeness, sub=Coverage Adequacy
- 5: tie_n=236, tie_nz=0.0042, non_tie_nz=0.5811, delta=0.5769, dim=clarity_and_communication, sub=Information Efficiency
- 49: tie_n=55, tie_nz=0.0000, non_tie_nz=0.5735, delta=0.5735, dim=helpfulness_and_usefulness, sub=Relevance to User
- 9: tie_n=40, tie_nz=0.0250, non_tie_nz=0.5893, delta=0.5643, dim=clarity_and_communication, sub=Tone and Register Match
- 51: tie_n=383, tie_nz=0.0131, non_tie_nz=0.5672, delta=0.5542, dim=relevance_instruction_following, sub=Task Adherence
- 52: tie_n=401, tie_nz=0.0125, non_tie_nz=0.5501, delta=0.5376, dim=relevance_instruction_following, sub=Task Adherence
- 53: tie_n=332, tie_nz=0.0120, non_tie_nz=0.5399, delta=0.5279, dim=relevance_instruction_following, sub=Task Adherence
- 47: tie_n=392, tie_nz=0.0051, non_tie_nz=0.5268, delta=0.5217, dim=helpfulness_and_usefulness, sub=Relevance to User
