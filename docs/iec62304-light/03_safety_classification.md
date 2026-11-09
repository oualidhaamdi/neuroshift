# 3. Software safety classification (hypothetical)

> Educational exercise — not regulatory documentation. See [README](README.md).

## 3.1 Hypothetical intended use

> *NeuroShift quantifies post-operative brain shift from a pre-operative and a
> post-operative T1-weighted MRI of adult patients who underwent brain tumour
> resection. Its output (displacement maps and regional statistics) is reviewed by
> a neuroradiologist or neurosurgeon **retrospectively**, to support the
> assessment of how much the anatomy moved relative to the pre-operative plan.
> It is not used intra-operatively and does not drive any device.*

Real intended use today: **none** — research software only.

## 3.2 Classification (IEC 62304 §4.3)

| Question | Answer |
|---|---|
| Can a failure of the software contribute to a hazardous situation? | Yes — an underestimated or mislocalised shift could mislead the clinician's assessment (see [04](04_risk_analysis.md)). |
| Do external risk control measures reduce the risk? | Partly — the output is reviewed by a qualified clinician alongside the original images, and it is not used for real-time guidance. |
| Worst credible harm after external controls | Non-serious injury (e.g. a delayed or additional follow-up imaging study, a suboptimal decision during later treatment planning that is caught at clinical review). |
| Can serious injury or death result? | Not under the intended use above. |

**Resulting class: B** — the software can contribute to a hazardous situation that
may result in *non-serious* injury.

## 3.3 When the class would change

- **Class C** if the output were used **intra-operatively** to update
  neuronavigation, or to set resection margins without independent review: an
  error could then contribute to serious injury (e.g. damage to eloquent cortex).
- **Class A** if the output were used only for research statistics with no
  influence on any individual patient's care.

## 3.4 Regulatory context (indicative)

Under the EU MDR, software that provides information used to take decisions with
diagnostic or therapeutic purposes falls under Annex VIII **Rule 11**, typically
**class IIa** or higher. Under the hypothetical intended use above, a manufacturer
would also need ISO 13485 QMS, IEC 62366-1 usability engineering, and clinical
evaluation — none of which exist for this research project.

## 3.5 Segregation

No segregation between items of different classes is claimed: the whole software
system is treated as class B. The validation framework (`src/qc/`) and the viewer
are treated as the same class because they produce results that a user could rely on.
