**Data Source**
This `v3` checklist bank was redesigned using remote-server evidence from:

- `/root/autodl-tmp/Thesis/results/checkeval_judge_dev_600_predictions.parquet`
- `/root/autodl-tmp/Thesis/results/ablation/ablation_drop_instruction_following_dev_600_question_diagnostics.csv`
- `/root/autodl-tmp/Thesis/results/checkeval_judge_dev_600_predictions_audit/*`

The server-side analysis was run inside:

- `/root/autodl-tmp/Thesis/.venvmerge`

**Selection Rules**
Questions were pruned or moved based on three observed failure modes:

1. `High same-rate`
   Questions where both responses usually received the same answer contributed little pairwise signal.
2. `High NA tendency`
   Questions with high per-response NA rates were usually conditional questions being asked too broadly.
3. `Conditional instruction checks mixed into correctness`
   Format, transformation, and explicit-constraint checks were useful when triggered, but noisy when left inside the shared correctness bank.

**Key Server Findings**

Mean same-rate by dimension on `dev_600`:

| dimension | mean same-rate | mean both-NA-rate |
|---|---:|---:|
| `clarity_and_communication` | 0.798 | 0.000 |
| `coding_communication_conditional` | 0.620 | 0.000 |
| `correctness_and_completeness` | 0.673 | 0.002 |
| `helpfulness_and_usefulness` | 0.664 | 0.001 |

This is why `clarity` was compressed aggressively in `v3`.

Examples of low-signal questions removed from the shared bank:

- `Is the response written clearly enough...`
- `Does the response use unambiguous wording...`
- `Are sentences phrased simply enough to be understood on a first read?`
- `Are the main points stated directly rather than implied vaguely?`

These regularly had same-rates around `0.90+` on the server.

Examples of retained or emphasized questions with materially better discriminative behavior:

- `Does the response provide enough explanation for the user to act on the answer without leaving important points implicit?`
- `Does the response resolve the user's central question fully enough...`
- `Does the response include enough underlying reasoning...`
- `Does the response explain the consequences or implications of following the advice?`
- `Does the response include specific steps, methods, or procedures...`
- `Does the response present the code as part of solving the user's specific request...`

Examples of questions moved into `relevance_instruction_following` because they were conditional but still useful:

- `Does the response follow the requested format or structure when one is explicitly specified?`
- `If the user specifies constraints on the requested output, does the response carry out the task within those constraints...?`
- `If the user requested an action on given content, does the response actually transform or use that content as requested?`
- `Does the response perform the task the user actually asked for...?`

**Design Outcome**

`v3` therefore does three things:

1. Removes high same-rate readability/style questions from the shared bank.
2. Keeps more discriminative completeness, helpfulness, and code-usefulness questions.
3. Adds a dedicated `relevance_instruction_following` module so instruction-sensitive checks can be selected explicitly instead of polluting the shared correctness bank.
