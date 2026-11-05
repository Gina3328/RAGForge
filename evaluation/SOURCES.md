# Evaluation documents

The evaluation documents are not redistributed in this repository. Download them
yourself and put them where the scripts expect them:

- the original three documents go in `evaluation/documents/`;
- the seven additional documents go in `evaluation/documents_new/`
  (the exact file names are listed in `evaluation/new_docs.tsv`).

| Key | Expected file name | Source |
|---|---|---|
| ccby | `ccby_4.0_legalcode.pdf` | [Creative Commons Attribution 4.0 International, legal code](https://creativecommons.org/licenses/by/4.0/legalcode) |
| cdc | `cdc_program_evaluation_framework_2024.pdf` | [CDC Program Evaluation Framework, 2024 (MMWR Recommendations and Reports, Vol. 73, No. RR-6)](https://www.cdc.gov/mmwr/volumes/73/rr/rr7306a1.htm) ([PDF](https://cdc.gov/mmwr/volumes/73/rr/pdfs/rr7306a1-H.pdf)) |
| ragbench | `ragbench_2407.11005v2.pdf` | [RAGBench, arXiv:2407.11005v2](https://arxiv.org/abs/2407.11005v2) |
| attention | `Attention Is All You Need.pdf` | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| nist_ai_rmf | `NIST.AI.100-1.pdf` | [NIST AI 100-1, Artificial Intelligence Risk Management Framework (AI RMF 1.0), January 2023](https://doi.org/10.6028/NIST.AI.100-1) |
| ipcc_spm | `IPCC_AR6_SYR_SPM.pdf` | [IPCC AR6 Synthesis Report, Summary for Policymakers](https://report.ipcc.ch/ar6syr/pdf/IPCC_AR6_SYR_SPM.pdf) |
| alice | `Alice_in_Wonderland.pdf` | "Alice's Adventures in Wonderland" (1865 first-edition digital replica by BookVirtual / VolumeOne Publishing, file version v1.2, 105 pages). The text is public domain. The original download location was not recorded. |
| who_handhygiene | `WHO_Hand_Hygiene_Guidelines_excerpt_Parts_II-IV.pdf` | Excerpt (Parts II-IV) of the [WHO guidelines on hand hygiene in health care, 2009](https://www.who.int/publications/i/item/9789241597906) |
| roomba | `Roomba® 600 Series Owner's Guide.pdf` | iRobot, Roomba 600 Series Owner's Guide (available from iRobot support). |
| tootsie | `Tootsie Roll Industries 2024 Annual Report.pdf` | Tootsie Roll Industries, 2024 Annual Report (63 pages). The FY2024 report is also filed with the SEC (EDGAR CIK 98677). The original download location was not recorded. |

Notes:

- `new_docs.tsv` matches the Roomba and Tootsie files with a `*` pattern, so a
  slightly different file name still works as long as the words match.
- The WHO guidelines are published with "All rights reserved". Only the
  question files, which contain short reference answers and evidence sentences,
  are included in this repository, not the document itself.
- If a document is replaced by a different edition, the chunk ids in the
  `reference_chunk_ids` fields of the question files may no longer match.
  Re-run `evaluation/list_chunks.py` and update them.
