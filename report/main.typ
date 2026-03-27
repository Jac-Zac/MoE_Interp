#import "@preview/red-agora:0.1.2": project

#show: project.with(
  title: "Expert Pursuit: Probing Expert Specialization in MoE Models",
  subtitle: "Analysis of Expert Specialization in MoE Transformer LLMs",
  authors: (
    "Jacopo Zacchigna",
  ),
  school-logo: [],
  company-logo: [],
  mentors: (
    "Prof. Alberto Cazzaniga",
  ),
  footer-text: "DSAI",
  branch: "NLP & Advanced Deep Learning",
  academic-year: "2025-2026",)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)


#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/methods.typ"
#include "sections/results.typ"
#include "sections/conclusion.typ"

= References
#bibliography("refs.bib")

#include "sections/appendix.typ"
