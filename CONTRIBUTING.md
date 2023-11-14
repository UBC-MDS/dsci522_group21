# Contributing to the Bank Marketing Campaign Analysis Project

This document outlines how to contribute to the Bank Marketing Campaign Analysis project.

> **This document aims to establish transparent expectations for all participants in this project, fostering a collective effort towards improvement. Adhering to these guidelines contributes to a constructive and inclusive environment, promoting a gratifying experience for both contributors annd maintainers.**

### Fixing typos

Small typos or grammatical errors in documentation may be corrected directly using the GitHub web interface, provided the changes are made in the _source_ file.

*NOTE*: you may not edit an `.md` file under `main`.

### Prerequisites

Before making a substantial pull request, please open an issue to discuss the proposed change. For bug reports, please provide a minimal reproducible example.

### Pull request process

*  Create a new Git branch for each pull request (PR).
*  New code should adhere to the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/).
*  Include tests for new funcitonality.
*  Update documentation accordingly.

### Writing Commit Messages

Please [write a great commit message](https://chris.beams.io/posts/git-commit/).

1. Separate subject from body with a blank line
2. Limit the subject line to 50 characters
3. Capitalize the subject line
4. Do not end the subject line with a period
5. Use the imperative mood in the subject line (example: "Fix networking issue")
6. Wrap the body at about 72 characters
7. Use the body to explain why, not what and how (the code shows that!)
8. If applicable, prefix the title with the relevant component name. (examples: "[Docs] Fix typo", "[Profile] Fix missing avatar")

```
[TAG] Short summary of changes in 50 chars or less

Add a more detailed explanation here, if necessary. Possibly give 
some background about the issue being fixed, etc. The body of the 
commit message can be several paragraphs. Further paragraphs come 
after blank lines and please do proper word-wrap.

Wrap it to about 72 characters or so. In some contexts, 
the first line is treated as the subject of the commit and the 
rest of the text as the body. The blank line separating the summary 
from the body is critical (unless you omit the body entirely); 
various tools like `log`, `shortlog` and `rebase` can get confused 
if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how or what. The code explains 
how or what. Reviewers and your future self can read the patch, 
but might not understand why a particular solution was implemented.
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

 - Bullet points are okay, too

 - A hyphen or asterisk should be used for the bullet, preceded
   by a single space, with blank lines in between

Note the fixed or relevant GitHub issues at the end:

Resolves: #123
See also: #456, #789
```

### Code of Conduct

Participation in this project requires compliance with our [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

### Attribution

These contributing guidelines were inspired by the [Breast_Cancer_Predictor_Project]
(https://github.com/ttimbers/breast_cancer_predictor).
