# Contributing

Contributions that improve reproducibility, portability, tests, or
documentation are welcome.

Before opening a pull request:

1. keep method changes separate from generated experiment outputs;
2. do not commit datasets, checkpoints, prediction archives, private paths, or
   credentials;
3. run `python -m unittest discover -s selector/tests`;
4. run the relevant DomainBed or appendix-baseline tests when dependencies are
   available;
5. explain any change to metric definitions, candidate eligibility, or
   tie-breaking because these can alter reported selections.

Large experiment outputs should be published through an external archival
repository with checksums, not committed to Git.
