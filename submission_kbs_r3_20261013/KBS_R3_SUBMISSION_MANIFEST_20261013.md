# KBS R3 final submission package manifest

**Prepared:** 2026-10-13  
**Manuscript:** KNOSYS-D-26-07461R2 — Decision-aligned eviction-value prediction for learning-augmented caching  
**Branch / audited commit:** `kbs/r3-manuscript-corrections-20261012` / `b7c2dd4707a387fd2e76b84b7a05a67df00c61e3`  
**Master ZIP:** `/home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/final_submission_package_20261013/KBS_R3_FINAL_SUBMISSION_PACKAGE_20261013.zip` (213362 bytes; SHA-256 `990228eee732d95d1b92a0cdf115b26a6571959325d9a78d1c6fdcbc6c61d401`)
**Master ZIP members:** exactly the seven files below; no manifest, author agreement, or historical artifact is inside. The manuscript PDF and embedded LaTeX ZIP are byte-identical to the prior validated package; among the seven members only the response DOCX changed.

| Archive filename | Bytes | SHA-256 | Source path |
|---|---:|---|---|
| `01_Revised_Manuscript.pdf` | 122471 | `6857ff08f66bebfc990d8e5bbed03347ba427898f6d7282d6430c7615a18eaed` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/main.pdf |
| `02_Response_to_Reviewers_R3.docx` | 14138 | `5a84fd1676796c23a157b3c9139dc2c79632b55e00c4267d068b4c73ec756067` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/02_Response_to_Reviewers_R3.docx |
| `03_Cover_Letter_R3.docx` | 11433 | `84086f5b1a8208b07bbb5ab091918eebda5399d67c6f0596d8dada9b0f6f082b` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/03_Cover_Letter_R3.docx |
| `04_Highlights_R3.docx` | 10963 | `d275f27ffc35aff0fdf87f6816b0ffbc51c7a8cdcda97ed13bf79d8b84136f92` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/04_Highlights_R3.docx |
| `05_CRediT_Author_Statement.docx` | 10732 | `7a0cb19c7c1cf86b1f81e7b9c731bae539a8a1136f8515f9da1aa1302d76afda` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/05_CRediT_Author_Statement_R3.docx |
| `06_Declaration_of_Interest.docx` | 10669 | `c71de6a921b10f003ef4744527986d114d7f914af64fa8a9b5cb4ac872e1a178` | /home/soroush/projects/augmented-caching/worktrees/kbs-r3-manuscript-corrections/r3_manuscript/06_Declaration_of_Interest_R3.docx |
| `07_R3_LaTeX_Source.zip` | 41533 | `ad79c6fb5533cb9ae8cab67a683298841fba265712d23cd43bd73ef2c2861d8e` | Fresh clean build of audited `r3_manuscript/main.tex`, `refs.bib`, `elsarticle.cls`, and `elsarticle-num.bst`; generated `main.bbl` |

## LaTeX source ZIP

Flat archive; all member paths are relative. The `.bbl` was regenerated from the audited manuscript and bibliography and matches the previously audited `.bbl` byte for byte.

| Member | Bytes | SHA-256 |
|---|---:|---|
| `main.tex` | 40057 | `2e4240255e869b51bdd4923192a526dcde52a7ee225a4ee03d06c308e16dc16e` |
| `refs.bib` | 19307 | `418db5c6c07f381861d0986c3967384eb6400b319e240d8c10c130ad8050dc03` |
| `elsarticle.cls` | 45216 | `ba7a1f25626a79fb39b53888dd826c4ae14d20db5429e236abdaaaf2fdf897ab` |
| `elsarticle-num.bst` | 29462 | `08d06d24a533fbe0c6e29d5bd5e7134781ab41a3c6640d73ad63112d4851b301` |
| `main.bbl` | 13005 | `a4ef496be786dd7655fc7ea3eeb00630359839caabdd87a926fbacb6d8da4ba2` |

## Validation

- **Clean-room compilation: PASS.** The embedded source ZIP was extracted from the master ZIP into a separate temporary directory and compiled with `tectonic main.tex --keep-logs --keep-intermediates`. The generated `.bbl` remained byte-identical to the packaged `.bbl`.
- **Manuscript: PASS.** Independently compiled and packaged PDFs are both 21 letter-size pages and have identical `pdftotext -layout` output. The packaged PDF is byte-identical to the audited R3 `main.pdf`.
- **Scientific content unchanged: YES.** Packaged `main.tex` is byte-identical to `HEAD:r3_manuscript/main.tex`; the response and other DOCX files are byte-identical to their audited R3 source documents. No optional experiment was run or result added.
- **Structure: PASS.** Eight tables, one editable TikZ workflow figure, four numbered equation environments; all table/figure labels are referenced. Citation keys and cross-reference targets resolve.
- **Reviewer response: PASS.** The four mandatory October responses are text-identical to the prior validated R3 DOCX. The optional-experiments section uses the approved wording, acknowledges delayed multi-seed and controlled EV timing work, and claims no new result. The corrected DOCX and editable Markdown have equivalent text. LibreOffice rendered the DOCX to a three-page letter-size PDF; all three pages were visually inspected for legibility, page numbering, heading color, spacing, and layout.
- **Archive integrity: PASS.** Both ZIPs passed CRC checks; each DOCX passed ZIP integrity checks and was readable by Pandoc. Master ZIP contains exactly seven top-level members and no recursive ZIP entry.
- **Original R2 artifacts: unchanged.** The correction updates only the R3 response source and DOCX, this ZIP, and this manifest.

## Outstanding submission actions and warnings

- The author must complete the official author agreement separately; it is intentionally absent.
- The decision checklist requests a revised-manuscript DOCX while permitting LaTeX source separately. The package supplies editable LaTeX source and PDF, with no faithful manuscript DOCX. Confirm that Editorial Manager accepts the LaTeX manuscript path, or prepare and inspect a faithful editable DOCX before upload.
- The build succeeds with inherited layout box warnings and three BibTeX metadata warnings: missing pages for `shah2022lodl` and `yang2023s3fifo`, and missing year for `wikimedia_pageviews`. No citation is unresolved.
- This package is for author review; it has not been submitted to Editorial Manager.
