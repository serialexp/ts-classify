# Changelog

## 0.1.1 (2026-02-10)

### Bug Fixes

- update README with correct npm package name and usage

### Chores

- add just-release workflows and CI setup
- remove old JS library/tests, rename and update license
- run tests in release workflow, remove from publish
- update just-release to 0.8.0
- update just-release to 0.8.1

### Other

- Initial commit
- tiny tweaks for docs
- Fixed predictOne function.
- Implemented "toJSON" and "fromJSON" functions
- make svm.js usable as a node module
- add package.json file for npm packaging
- add note about node.js usage to readme
- tiny tweaks, fixed bug in docs
- Big release! fromJSON and toJSON now work. Quite substantial efficiency improvements: the non-support vectors are pruned during training. Also, for linear SVM the weights are automatically computed and used which should be much faster than before. Slight API changes to train(), but backwards compatible.
- add 'memoize' option to cache kernel computations
- Rewrite SVM library from JavaScript to Rust with WASM support

